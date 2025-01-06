import os
from pathlib import Path
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from optimi import AdamW as OptimiAdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import multiprocessing
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration parameters
config = {
    "model_name": "microsoft/Florence-2-base",
    "dataset_path": "",
    "run_name": "",
    "epochs": 2,  # I found 3 or more to start overfitting. 1 or 2 is a good default.
    "learning_rate": 1e-5,
    "gradient_checkpointing": True,  # May have no effect
    "freeze_vision": True,
    "freeze_language": False,
    "freeze_other": False,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 16,
    "clip_grad_norm": 1,
    "weight_decay": 1e-5,  # 1e-5 default. Not sure if it should be higher or lower.
    "save_steps": 50,
    "save_total_limit": 3,
    "eval_steps": 50,
    "warmup_steps": 50
}


class LocalImageTextDataset(Dataset):
    def __init__(self, folder_path, task_prompt="<CAPTION>"):  # Change to whatever
        self.folder_path = Path(folder_path)
        self.data = [
            (file, file.with_suffix(".txt"))
            for file in self.folder_path.iterdir()
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and file.with_suffix(".txt").exists()
        ]
        self.task_prompt = task_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, prompt_path = self.data[idx]
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")  # Ensure the image has 3 channels
            with open(prompt_path, "r") as f:
                answer = f.read().strip()
            return self.task_prompt, answer, image
        except Exception as e:
            raise RuntimeError(f"Error loading data from {image_path}: {e}")


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


def train_model(train_loader, val_loader, model, processor, config):
    optimizer = OptimiAdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        decouple_lr=True
    )
    tokenizer = processor.tokenizer

    # Calculate total training steps
    total_training_steps = (len(train_loader) // config["gradient_accumulation_steps"]) * config["epochs"]
    if len(train_loader) % config["gradient_accumulation_steps"] != 0:
        total_training_steps += 1

    # Linear Warmup Scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-10,
        end_factor=1.0,
        total_iters=config["warmup_steps"]
    )

    # Cosine Annealing Warm Restarts Scheduler
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=(total_training_steps - config["warmup_steps"]) + 1,  # Add 1 to prevent it from spiking on the last step
    )

    # Combine schedulers using SequentialLR
    if config["warmup_steps"] > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config["warmup_steps"]]
        )
    else:
        scheduler = cosine_scheduler

    current_step = 0

    with wandb.init(project="Florence-2", name=config["run_name"]) as run:
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
            optimizer.zero_grad() # Initialize gradients outside the inner loop

            for i, batch in enumerate(train_loader):
                torch.cuda.empty_cache()
                inputs, answers = batch
                labels = tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    current_step += 1
                    progress_bar.update(1)

                    run.log(
                        {
                            "train/loss": loss.item() * config["gradient_accumulation_steps"],
                            "lr": optimizer.param_groups[0]["lr"]
                        }
                    )
                    progress_bar.set_postfix({"loss": loss.item() * config["gradient_accumulation_steps"]})

                    if current_step % config["eval_steps"] == 0:
                        evaluate_model(val_loader, model, processor, run, current_step)

                    if current_step % config["save_steps"] == 0:
                        save_model_checkpoint(model, processor, config["run_name"], current_step, config["save_total_limit"])

        # Save the last checkpoint
        save_model_checkpoint(model, processor, config["run_name"], current_step, config["save_total_limit"])


def evaluate_model(val_loader, model, processor, run, current_step):
    model.eval()
    tokenizer = processor.tokenizer
    total_loss = 0
    steps = 0
    table = wandb.Table(columns=["Ground Truth", "Prediction"])
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            torch.cuda.empty_cache()
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
                print("  GT:", answer)
                print("")
                print("Pred:", parsed_answer["<CAPTION>"].replace("<pad>", ""))
                print("----")

            break  # Only compare the first batch

    avg_loss = total_loss / steps
    run.log({"validation/avg_loss": avg_loss, "validation/predictions": table}, step=current_step)


def save_model_checkpoint(model, processor, run_name, step, save_total_limit):
    output_dir = f"./checkpoints/{run_name}/step-{step}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Implement save_total_limit
    checkpoint_dir = Path(f"./checkpoints/{run_name}")
    checkpoints = sorted(checkpoint_dir.glob("step-*"), key=lambda x: int(x.name.split('-')[-1]))
    if len(checkpoints) > save_total_limit:
        num_to_delete = len(checkpoints) - save_total_limit
        for checkpoint_to_delete in checkpoints[:num_to_delete]:
            print(f"Deleting old checkpoint: {checkpoint_to_delete}")
            shutil.rmtree(checkpoint_to_delete)


# Initialize components
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
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

train_dataset = LocalImageTextDataset(f"{config['dataset_path']}/train")
val_dataset = LocalImageTextDataset(f"{config['dataset_path']}/eval")

train_loader = DataLoader(
    train_dataset,
    batch_size=config["train_batch_size"],
    collate_fn=lambda batch: collate_fn(batch, processor),
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["eval_batch_size"],
    collate_fn=lambda batch: collate_fn(batch, processor),
)

# Train the model
train_model(train_loader, val_loader, model, processor, config)
