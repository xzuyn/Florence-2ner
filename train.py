import os
import gc
import json
import random
import multiprocessing
from pathlib import Path
import shutil

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
    _LRScheduler
)
from transformers import AutoModelForCausalLM, AutoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration parameters
config = {
    "model_name": "microsoft/Florence-2-base",
    "task_prompt": "<CAPTION>",
    "dataset_path": "",
    "wandb_project_name": "Florence-2-base",
    "run_name": "",
    "epochs": 1,  # I found 3 or more to start overfitting. 1 or 2 is a good default
    "optimizer": "CAME",  # Currently supports "OptimiAdamW" & "CAME"
    "learning_rate": 1e-6,
    "lr_scheduler": "REX",  # Currently supports "Constant", "Cosine", and "REX"
    "gradient_checkpointing": True,  # May have no effect
    "freeze_vision": False,
    "freeze_language": False,
    "freeze_other": False,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 32,
    "clip_grad_norm": 1,
    "weight_decay": 1e-2,  # 1e-5 default for OptimiAdamW, 1e-2 default for CAME
    "save_total_limit": 3,
    "save_steps": 50,
    "eval_steps": 50,
    "warmup_steps": 50,
    "eval_split_ratio": 0.1,
    "seed": 42,
    "filtering_processes": 128,
    "attn_implementation": "sdpa"
}


class LocalImageTextDataset(Dataset):
    def __init__(self, data_pairs, task_prompt=config["task_prompt"]):
        self.data_pairs = data_pairs
        self.task_prompt = task_prompt

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, prompt_path = self.data_pairs[idx]
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            with open(prompt_path, "r") as f:
                answer = f.read().strip()
            return self.task_prompt, answer, image
        except Exception as e:
            raise RuntimeError(f"Error loading data from {image_path}: {e}")


# https://github.com/IvanVassi/REX_LR
class RexLR(_LRScheduler):
    def __init__(self, optimizer, max_val, min_val, num_epochs=1, last_epoch=-1):
        self.num_epochs = num_epochs
        self.min_val = min_val
        self.max_val = max_val
        if not self.min_val <= self.max_val:
            raise ValueError('Value of "min_val" should be less '
                             ' than value of "max_val". Got min_val=' + str(min_val) + ' and max_val=' + str(max_val))

        super(RexLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        mod_iter = float(self.last_epoch % self.num_epochs)
        z = float(self.num_epochs - mod_iter) / self.num_epochs
        val = self.min_val + float(self.max_val - self.min_val) * (z / (1 - 0.9 + 0.9 * z))
        return [val]


def collate_fn(batch, processor):
    questions, answers, images = zip(*batch)

    # Convert all images to RGB if needed
    validated_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    try:
        inputs = processor(
            text=list(questions),
            images=validated_images,
            return_tensors="pt",
            padding=True,
        ).to(device, torch.bfloat16)
    except ValueError as e:
        print(f"Processor error: {e}")
        raise

    return inputs, list(answers)


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
            weight_decay=optimizer_weight_decay
        )
    else:
        optimizer = OptimiAdamW(
            model_parameters,
            lr=optimizer_lr,
            weight_decay=optimizer_weight_decay,
            decouple_lr=True
        )

    return optimizer


# TODO: Add more LR schedulers
# TODO: Properly solve learning rate spiking on last few steps
def prepare_lr_scheduler(
    scheduler_optimizer,
    scheduler_name,
    scheduler_lr,
    scheduler_warmup_steps,
    scheduler_total_training_steps
):
    warmup_scheduler = LinearLR(
        scheduler_optimizer,
        start_factor=1e-20,
        end_factor=1.0,
        total_iters=scheduler_warmup_steps
    )

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
            scheduler_optimizer,
            max_val=scheduler_lr,
            min_val=0,
            num_epochs=(scheduler_total_training_steps - scheduler_warmup_steps) + 1
        )
    else:
        print("No valid LR scheduler selected. Falling back to Cosine.")
        main_scheduler = CosineAnnealingLR(
            scheduler_optimizer,
            T_max=(scheduler_total_training_steps - scheduler_warmup_steps) + 1
        )

    if scheduler_warmup_steps > 0:
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
    total_training_steps = (len(train_loader) // config["gradient_accumulation_steps"]) * config["epochs"]
    if len(train_loader) % config["gradient_accumulation_steps"] != 0:
        total_training_steps += 1

    scheduler = prepare_lr_scheduler(
        optimizer,
        config["lr_scheduler"],
        config["learning_rate"],
        config["warmup_steps"],
        total_training_steps
    )

    current_step = 0
    with wandb.init(project=config["wandb_project_name"], name=config["run_name"]) as run:
        evaluate_model(val_loader, model, processor, run, current_step)  # Evaluate at the beginning

        for epoch in range(config["epochs"]):
            model.train()
            progress_bar = tqdm(
                range(
                    len(train_loader) // config["gradient_accumulation_steps"]
                    if len(train_loader) % config["gradient_accumulation_steps"] == 0
                    else len(train_loader) // config["gradient_accumulation_steps"] + 1
                ),
                desc=f"Training [Epoch {epoch + 1}/{config['epochs']}]"
            )
            optimizer.zero_grad()  # Initialize gradients outside the inner loop

            for i, batch in enumerate(train_loader):
                inputs, answers = batch
                labels = tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                # Create attention mask to ignore padding tokens
                attention_mask = labels != tokenizer.pad_token_id

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
                    torch.cuda.empty_cache()
                    gc.collect()

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    current_step += 1
                    progress_bar.update(1)

                    run.log(
                        {
                            "train/loss": loss.item() * config["gradient_accumulation_steps"],
                            "train/grad_norm": grad_norm.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": current_step / (total_training_steps / config["epochs"])
                        }
                    )
                    progress_bar.set_postfix(
                        {
                            "loss": loss.item() * config["gradient_accumulation_steps"],
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
    tokenizer = processor.tokenizer
    total_loss = 0
    steps = 0
    table = wandb.Table(columns=["Ground Truth", "Prediction"])
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            torch.cuda.empty_cache()
            gc.collect()
            inputs, answers = batch
            labels = tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(device)

            # Create attention mask for padding tokens
            attention_mask = labels != tokenizer.pad_token_id

            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                labels=labels,
                attention_mask=attention_mask
            )
            total_loss += outputs.loss.item()
            steps += 1

        for batch in val_loader:
            inputs, answers = batch
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=5,
                do_sample=True
            )
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )

            for generated_text, answer in zip(generated_texts, answers):
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task="<CAPTION>",
                    image_size=(
                        inputs["pixel_values"].shape[-2],
                        inputs["pixel_values"].shape[-1],
                    ),
                )
                # Log to wandb table
                table.add_data(answer, parsed_answer["<CAPTION>"].replace("<pad>", ""))
                print("\n----")
                print("Ground Truth:", answer)
                print("")
                print("  Prediction:", parsed_answer["<CAPTION>"].replace("<pad>", ""))
                print("----")

            break  # Only compare the first batch

    avg_loss = total_loss / steps
    run.log({"validation/avg_loss": avg_loss, "validation/predictions": table}, step=current_step)


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
    filtered_chunk = []
    tokenizer = processor.tokenizer
    for img_path, txt_path in chunk:
        try:
            with open(txt_path, "r") as f:
                text = f.read().strip()
            inputs = tokenizer(text, return_tensors="pt")
            if inputs.input_ids.shape[1] <= 1000:  # TODO: Properly calculate (1024 - task prompt token count)
                filtered_chunk.append((img_path, txt_path))
        except Exception as e:
            print(f"Error processing {txt_path}: {e}")
    return filtered_chunk


verify_optimizer_choice(config["optimizer"])

# Initialize components
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation=config["attn_implementation"],
).to(device)
processor = AutoProcessor.from_pretrained(config["model_name"], trust_remote_code=True)

if config["gradient_checkpointing"]:
    model.gradient_checkpointing_enable()

if config["freeze_language"]:
    for param in model.language_model.parameters():
        param.requires_grad = False

if config["freeze_vision"]:
    for param in model.vision_tower.parameters():
        param.requires_grad = False

if config["freeze_other"]:
    image_pos_embed.column_embeddings.weight.requires_grad = False
    image_pos_embed.row_embeddings.weight.requires_grad = False
    visual_temporal_embed.pos_idx_to_embed.requires_grad = False
    image_proj_norm.bias.requires_grad = False
    image_proj_norm.weight.requires_grad = False
    image_projection.requires_grad = False

random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])

dataset_path = Path(config["dataset_path"])
all_files = [
    (img_file, img_file.with_suffix(".txt")) for img_file in dataset_path.iterdir()
    if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and img_file.with_suffix(".txt").exists()
]

gc.collect()

# Optimized filtering using multiprocessing
print("Filtering data based on token length using multiprocessing...")
with multiprocessing.Pool(processes=config["filtering_processes"]) as pool:
    chunk_size = max(1, len(all_files) // config["filtering_processes"])
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    results = list(tqdm(pool.starmap(filter_data_chunk, [(chunk, processor) for chunk in chunks]), total=len(chunks)))

gc.collect()

filtered_files = [item for sublist in results for item in sublist]

print(f"Filtered out {len(all_files) - len(filtered_files)} files due to token length.")
all_files = filtered_files

random.shuffle(all_files)
eval_size = int(len(all_files) * config["eval_split_ratio"])
eval_dataset_pairs = all_files[:eval_size]
train_dataset_pairs = all_files[eval_size:]

train_dataset = LocalImageTextDataset(train_dataset_pairs)
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

torch.cuda.empty_cache()
gc.collect()

# Train the model
train_model(train_loader, val_loader, model, processor, config)
