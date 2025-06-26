import os
import gc
import json
import random
import multiprocessing
from pathlib import Path
import shutil
import re
import math
from PIL import Image
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LRScheduler
)
from transformers import AutoModelForCausalLM, AutoProcessor

# Allow extremely large images
Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configuration parameters
config = {
    "model_name": "microsoft/Florence-2-base",
    "wandb_project_name": "Florence-2-base",
    "run_name": "",
    "epochs": 2,  # I found 3 or more to start overfitting. 1 or 2 is a good default
    "optimizer": "CAME",  # Currently supports "OptimiAdamW" & "CAME"
    "learning_rate": 1e-6,
    "min_learning_rate": 1e-7,  # Currently only works with REX
    "lr_scheduler": "REX",  # Currently supports "Constant", "Cosine", and "REX"
    "gradient_checkpointing": True,  # May have no effect
    "freeze_vision": False,
    "freeze_language": False,
    "freeze_other": False,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "clip_grad_norm": 1,
    "weight_decay": 0.01,  # 1e-5 default for OptimiAdamW, 1e-2 default for CAME
    "save_total_limit": 3,
    "save_steps": 10,
    "eval_steps": 10,
    "warmup_steps": 50,
    "eval_split": 0.1,
    "seed": 42,
    "filtering_processes": 128,
    "attn_implementation": "sdpa"
    "dataset_config": {
        "<CAPTION>": [
            "...",
        ],
        # "<DETAILED_CAPTION>": [
        #     "...",
        # ]
    }
}


# https://github.com/IvanVassi/REX_LR
class RexLR(LRScheduler):
    """
    Reflected Exponential (REX) learning rate scheduler.

    - Original implementation: https://github.com/IvanVassi/REX_LR
    - Original license: Apache 2.0
    - Based on: https://arxiv.org/abs/2107.04197

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule the learning rate for.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.
        total_steps (int): The total number of training steps.
        num_warmup_steps (int): The number of warmup steps.
        last_step (int): The index of last step.
    """

    def __init__(
        self, optimizer, max_lr, min_lr, total_steps=0, num_warmup_steps=0, last_step=0
    ):
        if min_lr > max_lr:
            raise ValueError(
                f'Value of "min_lr" should be less than value of "max_lr". Got min_lr={min_lr} and max_lr={max_lr}'
            )
        if num_warmup_steps > total_steps:
            raise ValueError(
                f"num_warmup_steps ({num_warmup_steps}) must be less than or equal to total_steps ({total_steps})."
            )

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.last_step = max(last_step - 1, 0)

        # Ensure each parameter group has an "initial_lr" key to avoid issues when resuming.
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        # Pass self.last_step as last_epoch to the parent.
        super().__init__(optimizer, last_epoch=self.last_step)

    @property
    def last_step(self):
        return self.last_epoch

    @last_step.setter
    def last_step(self, value):
        self.last_epoch = value

    def get_lr(self):
        # Warmup phase: if defined, increase lr linearly from 0 to max_lr.
        if 1 <= self.last_step <= self.num_warmup_steps:
            return [
                base_lr * self.last_step / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]

        # Post-warmup phase: adjust step relative to the end of warmup.
        step_after = self.last_step - self.num_warmup_steps
        remaining_steps = self.total_steps - self.num_warmup_steps

        # Avoid LR spiking
        if step_after >= remaining_steps or step_after == -1 or remaining_steps <= 0:
            return [self.min_lr for _ in self.base_lrs]

        mod_iter = step_after % remaining_steps
        z = (remaining_steps - mod_iter) / remaining_steps
        rex_factor = self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * (
            z / (0.1 + 0.9 * z)
        )
        return [base_lr * rex_factor for base_lr in self.base_lrs]


class LocalImageTextDataset(Dataset):
    def __init__(self, data_pairs):
        # data_pairs: list of tuples (task_prompt, image_path, text_path)
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        task_prompt, image_path, prompt_path = self.data_pairs[idx]
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            with open(prompt_path, "r") as f:
                answer = f.read().replace("  ", " ").strip()  # Remove double spaces, and strip
            return task_prompt, answer, image
        except Exception as e:
            raise RuntimeError(f"Error loading data from {image_path}: {e}")


def get_all_files_by_prompt(dataset_config):
    """Gather all (prompt, image, text) tuples from multiple directories per task."""
    all_pairs = []
    for task_prompt, dirs in dataset_config.items():
        for d in dirs:
            for root, _, files in os.walk(d):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        img_file = Path(root) / file
                        txt_file = img_file.with_suffix(".txt")
                        if txt_file.exists():
                            all_pairs.append((task_prompt, img_file, txt_file))
    return all_pairs


def collate_fn(batch, processor):
    # Unpack triples
    tasks, answers, images = zip(*batch)

    # Convert all images to RGB if needed
    validated_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    try:
        inputs = processor(
            text=list(tasks),
            images=validated_images,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(device, torch.bfloat16)
    except ValueError as e:
        print(f"Processor error: {e}")
        raise

    return list(tasks), inputs, list(answers)


# TODO: Add more optimizers
def verify_optimizer_choice(optimizer_choice):
    global OptimiAdamW, CAME

    if optimizer_choice == "OptimiAdamW":
        try:
            from optimi import AdamW as OptimiAdamW
        except ImportError:
            raise ImportError("You do not have optimī installed. Please install it using `pip install torch-optimi`")
    elif optimizer_choice == "CAME":
        try:
            from came_pytorch import CAME
        except ImportError:
            raise ImportError("You do not have CAME installed. Please install it using `pip install came-pytorch`")
    else:
        print("No valid optimizer selected. Falling back to OptimiAdamW.")
        try:
            from optimi import AdamW as OptimiAdamW
        except ImportError:
            raise ImportError("You do not have optimī installed. Please install it using `pip install torch-optimi`")


# TODO: Add more optimizers
# https://github.com/warner-benjamin/optimi
# https://github.com/yangluo7/CAME
def prepare_optimizer(
    model_parameters,
    optimizer_name,
    optimizer_lr,
    optimizer_weight_decay
):
    if optimizer_name == "OptimiAdamW":
        optimizer = OptimiAdamW(
            model_parameters,
            lr=optimizer_lr,
            weight_decay=optimizer_weight_decay,
            decouple_lr=True
        )
    elif optimizer_name == "CAME":
        optimizer = CAME(
            model_parameters,
            lr=optimizer_lr,
            weight_decay=optimizer_weight_decay,
        )
    else:
        optimizer = OptimiAdamW(
            model_parameters,
            lr=optimizer_lr,
            weight_decay=optimizer_weight_decay,
            decouple_lr=True,
        )

    return optimizer


# TODO: Add more LR schedulers
# TODO: Properly solve learning rate spiking on last few steps
def prepare_lr_scheduler(
    scheduler_optimizer,
    scheduler_name,
    scheduler_lr,
    scheduler_min_lr,
    scheduler_warmup_steps,
    scheduler_total_training_steps
):
    if scheduler_name == "Constant":
        main_scheduler = ConstantLR(
            scheduler_optimizer
        )
    elif scheduler_name == "Cosine":
        main_scheduler = CosineAnnealingLR(
            scheduler_optimizer,
            T_max=(scheduler_total_training_steps - scheduler_warmup_steps) + 1
        )
    elif scheduler_name == "REX":
        main_scheduler = RexLR(
            optimizer=scheduler_optimizer,
            max_lr=scheduler_lr,
            min_lr=scheduler_min_lr,
            total_steps=scheduler_total_training_steps,  # - scheduler_warmup_steps,
            num_warmup_steps=scheduler_warmup_steps
        )
    else:
        print("No valid LR scheduler selected. Falling back to Cosine.")
        main_scheduler = CosineAnnealingLR(
            scheduler_optimizer,
            T_max=(scheduler_total_training_steps - scheduler_warmup_steps) + 1
        )

    if scheduler_warmup_steps > 0 and scheduler_name != "REX":
        warmup_scheduler = LinearLR(
            scheduler_optimizer,
            start_factor=1e-20,
            end_factor=1.0,
            total_iters=scheduler_warmup_steps
        )
        scheduler = SequentialLR(
            scheduler_optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[scheduler_warmup_steps]
        )
    else:
        scheduler = main_scheduler

    return scheduler


def train_model(
    train_loader,
    val_loader,
    model,
    processor,
    config
):
    tokenizer = processor.tokenizer
    optimizer = prepare_optimizer(
        model.parameters(),
        config["optimizer"],
        config["learning_rate"],
        config["weight_decay"]
    )

    # Calculate total training steps
    total_training_steps = math.ceil(len(train_loader) / config["gradient_accumulation_steps"]) * config["epochs"]
    if len(train_loader) % config["gradient_accumulation_steps"] != 0:
        total_training_steps += 1

    scheduler = prepare_lr_scheduler(
        optimizer,
        config["lr_scheduler"],
        config["learning_rate"],
        config["min_learning_rate"],
        config["warmup_steps"],
        total_training_steps
    )

    current_step = 0
    with wandb.init(project=config["wandb_project_name"], name=config["run_name"]) as run:
        evaluate_model(val_loader, model, processor, run, current_step)  # Evaluate at the beginning

        for epoch in range(config["epochs"]):
            model.train()
            progress_bar = tqdm(range(int(total_training_steps / config["epochs"])), desc=f"Training [Epoch {epoch + 1}/{config['epochs']}]")
            optimizer.zero_grad()  # Initialize gradients outside the inner loop

            for i, batch in enumerate(train_loader):
                _, inputs, answers = batch
                labels = tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                # Create attention mask to ignore padding tokens
                attention_mask = labels != tokenizer.pad_token_id

                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        labels=labels,
                        attention_mask=attention_mask
                    )
                    loss = outputs.loss
                    loss = loss / config["gradient_accumulation_steps"]
                loss.backward()

                if (i + 1) % config["gradient_accumulation_steps"] == 0 or (i + 1) == len(train_loader):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    current_step += 1
                    progress_bar.update(1)

                    scalar_loss = loss.detach().item() * config["gradient_accumulation_steps"]
                    run.log(
                        {
                            "train/loss": scalar_loss,
                            "train/grad_norm": grad_norm.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": current_step / (total_training_steps / config["epochs"])
                        }
                    )
                    progress_bar.set_postfix(
                        {
                            "loss": scalar_loss,
                            "grad_norm": grad_norm.item()
                        }
                    )

                    if current_step % config["eval_steps"] == 0:
                        evaluate_model(
                            val_loader,
                            model,
                            processor,
                            run,
                            current_step
                        )

                    if current_step % config["save_steps"] == 0:
                        save_model_checkpoint(
                            model,
                            processor,
                            config["run_name"],
                            current_step,
                            config["save_total_limit"]
                        )

        # Eval the last step if it hasn't been already
        if current_step % config["eval_steps"] != 0:
            evaluate_model(
                val_loader,
                model,
                processor,
                run,
                current_step
            )

        # Save the last checkpoint
        save_model_checkpoint(model, processor, config["run_name"], current_step, config["save_total_limit"])


def evaluate_model(
    val_loader,
    model,
    processor,
    run,
    current_step
):
    model.eval()
    total_loss = 0
    steps = 0
    table = wandb.Table(columns=["Ground Truth", "Prediction"])

    # Loss computation
    with torch.no_grad():
        for tasks, inputs, answers in tqdm(val_loader, desc="Validation"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding="longest",
                truncation=False,
            ).input_ids.to(device)

            # Create attention mask for padding tokens
            attention_mask = labels != processor.tokenizer.pad_token_id

            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels,
                    attention_mask=attention_mask
                )
                total_loss += outputs.loss.item()
            steps += 1

        # Sample predictions (first batch only)
        for tasks, inputs, answers in val_loader:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=5,
                min_p=0.05,
                do_sample=True
            )
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )

            for prompt, gen_text, answer in zip(tasks, generated_texts, answers):
                parsed = processor.post_process_generation(
                    gen_text,
                    task=prompt,
                    image_size=(
                        inputs["pixel_values"].shape[-2],
                        inputs["pixel_values"].shape[-1],
                    ),
                )
                # Log to wandb
                table.add_data(answer, parsed[prompt].replace("<pad>", ""))
                print(f"\n----\n      Prompt: {prompt}\nGround Truth: {answer}\n  Prediction: {parsed[prompt].replace('<pad>', '')}\n----")
            break

    avg_loss = total_loss / steps
    run.log(
        {
            "validation/avg_loss": avg_loss,
            "validation/predictions": table
        },
        step=current_step
    )


def save_model_checkpoint(
    model,
    processor,
    run_name,
    step,
    save_total_limit
):
    output_dir = f"./checkpoints/{run_name}/step-{step}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Workaround for vision_config
    with open(f"{output_dir}/config.json", "r") as f:
        data = json.load(f)
    data["vision_config"]["model_type"] = "davit"
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(data, f, indent=2)

    # Implement save_total_limit
    checkpoint_dir = Path(f"./checkpoints/{run_name}")
    checkpoints = sorted(checkpoint_dir.glob("step-*"), key=lambda x: int(x.name.split('-')[-1]))
    if len(checkpoints) > save_total_limit:
        num_to_delete = len(checkpoints) - save_total_limit
        for checkpoint_to_delete in checkpoints[:num_to_delete]:
            print(f"Deleting old checkpoint: {checkpoint_to_delete}")
            shutil.rmtree(checkpoint_to_delete)


def filter_data_chunk(chunk, processor):
    """Filter a chunk of (task_prompt, image_path, text_path) tuples by token-length."""
    filtered_chunk = []
    tokenizer = processor.tokenizer
    for task_prompt, img_path, txt_path in chunk:
        try:
            with open(txt_path, "r") as f:
                text = f.read().replace("  ", " ").strip()  # Remove double spaces, and strip
            inputs = tokenizer(text, return_tensors="pt")
            if inputs.input_ids.shape[1] <= 1000:  # TODO: Properly calculate (1024 - task prompt token count)
                filtered_chunk.append((task_prompt, img_path, txt_path))
            else:
                print(f"Caption too long: {img_path}")
        except Exception as e:
            print(f"Error processing {txt_path}: {e}")
    return filtered_chunk

# Initialize components
verify_optimizer_choice(config["optimizer"])
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation=config["attn_implementation"],
).to(device)
processor = AutoProcessor.from_pretrained(config["model_name"], trust_remote_code=True)

# TODO: Move all this to a function (enable_optimizations())
# TODO: Verify if this works, and if unsloth/OneTrainer CPU offloaded checkpointing can be added
if config["gradient_checkpointing"]:
    model.gradient_checkpointing_enable()
if config["freeze_language"]:
    for param in model.language_model.parameters():
        param.requires_grad = False
if config["freeze_vision"]:
    for param in model.vision_tower.parameters():
        param.requires_grad = False
if config["freeze_other"]:
    model.image_pos_embed.column_embeddings.weight.requires_grad = False
    model.image_pos_embed.row_embeddings.weight.requires_grad = False
    model.visual_temporal_embed.pos_idx_to_embed.requires_grad = False
    model.image_proj_norm.bias.requires_grad = False
    model.image_proj_norm.weight.requires_grad = False
    model.image_projection.requires_grad = False

random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])

# Gather and filter data
print("Gathering files from multiple datasets...")
all_pairs = get_all_files_by_prompt(config["dataset_config"])

gc.collect()

# Optimized filtering using multiprocessing
print("Filtering data based on token length using multiprocessing...")
with multiprocessing.Pool(processes=config["filtering_processes"]) as pool:
    chunk_size = max(1, len(all_pairs) // config["filtering_processes"])
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, len(all_pairs), chunk_size)]
    results = list(tqdm(
        pool.starmap(filter_data_chunk, [(chunk, processor) for chunk in chunks]),
        total=len(chunks)
    ))
filtered_pairs = [item for sublist in results for item in sublist]
print(f"Filtered out {len(all_pairs) - len(filtered_pairs)} files due to token length.")

random.shuffle(filtered_pairs)
eval_size = int(len(filtered_pairs) * config["eval_split"]) if config["eval_split"] < 1 else min(int(config["eval_split"]), len(filtered_pairs))
eval_dataset_pairs = filtered_pairs[:eval_size]
train_dataset_pairs = filtered_pairs[eval_size:]

train_dataset = LocalImageTextDataset(train_dataset_pairs)
# TODO: Add ability to specify a val set
val_dataset = LocalImageTextDataset(eval_dataset_pairs)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["train_batch_size"],
    collate_fn=lambda batch: collate_fn(batch, processor),
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["eval_batch_size"],
    collate_fn=lambda batch: collate_fn(batch, processor)
)

torch.cuda.synchronize()
torch.cuda.empty_cache()
gc.collect()

# Train the model
train_model(train_loader, val_loader, model, processor, config)
