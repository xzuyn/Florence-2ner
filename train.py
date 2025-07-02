import os
import gc
import re
import math
import wandb
import json
import yaml
import shutil
import random
import argparse
import multiprocessing

from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LRScheduler
)


# Allow extremely large images
Image.MAX_IMAGE_PIXELS = None

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        self,
        optimizer,
        max_lr,
        min_lr,
        total_steps=0,
        num_warmup_steps=0,
        last_step=0,
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
        rex_factor = self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * (z / (0.1 + 0.9 * z))

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
    """
    Gather all (prompt, image, text) tuples from multiple directories per task.
    """

    all_pairs = []
    for task_prompt, dirs in dataset_config.items():
        count = 0
        for d in dirs:
            for root, _, files in os.walk(d):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        img_file = Path(root) / file
                        txt_file = img_file.with_suffix(".txt")
                        if txt_file.exists():
                            all_pairs.append((task_prompt, img_file, txt_file))
                            count += 1

        print(f"{task_prompt}: found {count} pairs")

    return all_pairs


def filter_data_chunk(chunk, processor):
    tokenizer = processor.tokenizer

    texts = []
    meta = []
    for task_prompt, img_path, txt_path in chunk:
        try:
            with open(txt_path, "r") as f:
                text = f.read().replace("  ", " ").strip()
            texts.append(text)
            meta.append((task_prompt, img_path, txt_path))
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")

    if not texts:
        return []

    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    filtered = []
    for (task_prompt, img_path, txt_path), ids in zip(meta, tokenized.input_ids):
        length = len(ids)
        if length <= 1000:  # TODO: Properly calculate (1024 - task prompt token count)
            filtered.append((task_prompt, img_path, txt_path))
        else:
            print(f"Caption too long ({length} tokens): {img_path}")

    return filtered


def filter_all_pairs(all_pairs, processor, processes_count, batch_size=1):
    batches = [all_pairs[i : i + batch_size] for i in range(0, len(all_pairs), batch_size)]

    worker = partial(filter_data_chunk, processor=processor)

    results = []
    with multiprocessing.Pool(processes=processes_count, maxtasksperchild=None) as pool:
        with tqdm(total=len(batches)) as pbar:
            for chunk_result in pool.imap_unordered(worker, batches):
                results.extend(chunk_result)
                pbar.update()

    return results


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
        )
    except ValueError as e:
        print(f"Processor error: {e}")
        raise

    return list(tasks), inputs, list(answers)


def safe_forward(model, inputs, labels, attention_mask, max_chunks=4):
    """
    Tries to forward a batch. If OOM, retries with smaller chunks.
    """

    try:
        return model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            labels=labels,
            attention_mask=attention_mask
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise

        print("Got OOM. Trying chunked forward pass.")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        bsz = inputs["input_ids"].size(0)
        for num_chunks in range(2, max_chunks + 1):
            if bsz % num_chunks != 0:
                continue  # only support clean splits for now
            try:
                chunk_size = bsz // num_chunks
                losses = []

                for i in range(num_chunks):
                    sl = slice(i * chunk_size, (i + 1) * chunk_size)
                    input_chunk = {k: v[sl] for k, v in inputs.items()}
                    label_chunk = labels[sl]
                    mask_chunk = attention_mask[sl]

                    outputs = model(
                        input_ids=input_chunk["input_ids"],
                        pixel_values=input_chunk["pixel_values"],
                        labels=label_chunk,
                        attention_mask=mask_chunk
                    )
                    losses.append(outputs.loss.detach())

                outputs.loss = torch.stack(losses).mean()
                return outputs
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    print(f"Still OOM at {num_chunks} chunks. Trying smaller.")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise

        raise RuntimeError("OOM even after chunking. Try lowering batch size.")


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
        try:
            from optimi import AdamW as OptimiAdamW
            optimizer = OptimiAdamW(
                model_parameters,
                lr=optimizer_lr,
                weight_decay=optimizer_weight_decay,
                decouple_lr=True
            )
        except ImportError:
            raise ImportError("You do not have optimī installed. Please install it using `pip install torch-optimi`")
    elif optimizer_name == "CAME":
        try:
            from came_pytorch import CAME
            # TODO: Make optional
            optimizer = CAME(
                model_parameters,
                lr=optimizer_lr,
                weight_decay=optimizer_weight_decay,
                enable_stochastic_rounding=True,
                enable_cautious=True,
                enable_8bit=True,
            )
        except ImportError:
            raise ImportError(
                "You do not have CAME installed. "
                "Please install it using `pip install came-pytorch @ git+https://github.com/xzuyn/CAME.git@sr-grams-cautious-8bit`"
            )
    else:
        raise RuntimeError("No valid optimizer selected. Falling back to OptimiAdamW.")

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
            total_steps=scheduler_total_training_steps,
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
        # Evaluate before any training starts
        evaluate_model(val_loader, model, processor, run, current_step, config)

        for epoch in range(config["epochs"]):
            model.train()
            progress_bar = tqdm(range(int(total_training_steps / config["epochs"])), desc=f"Training [Epoch {epoch + 1}/{config['epochs']}]")
            optimizer.zero_grad()

            for i, (_, inputs, answers) in enumerate(train_loader):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

                labels = tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                    pad_to_multiple_of=16,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                # Create attention mask to ignore padding tokens
                attention_mask = labels != tokenizer.pad_token_id

                # Move inputs to device and cast pixel_values to bf16
                inputs = inputs.to(device)
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, cache_enabled=True):
                    outputs = safe_forward(
                        model=model,
                        inputs=inputs,
                        labels=labels,
                        attention_mask=attention_mask,
                        max_chunks=config["train_batch_size"]
                    )
                    loss = outputs.loss / config["gradient_accumulation_steps"]
                    del inputs, answers, labels, attention_mask, outputs

                loss.backward()

                if (i + 1) % config["gradient_accumulation_steps"] == 0 or (i + 1) == len(train_loader):
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

                    del loss, scalar_loss, grad_norm

                    if current_step % config["eval_steps"] == 0:
                        evaluate_model(val_loader, model, processor, run, current_step, config)

                    if current_step % config["save_steps"] == 0:
                        save_model_checkpoint(
                            model,
                            processor,
                            config["run_name"],
                            current_step,
                            config["save_total_limit"]
                        )

            if config["save_on_epoch_too"]:
                save_model_checkpoint(model, processor, config["run_name"], current_step, config["save_total_limit"])

        # Eval the last step if it hasn't been already
        if current_step % config["eval_steps"] != 0:
            evaluate_model(val_loader, model, processor, run, current_step, config)

        # Save the last checkpoint
        save_model_checkpoint(model, processor, config["run_name"], current_step, config["save_total_limit"])


def evaluate_model(
    val_loader,
    model,
    processor,
    run,
    current_step,
    config
):
    model.eval()
    total_loss = 0
    steps = 0
    table = wandb.Table(columns=["Ground Truth", "Prediction"])

    # Loss computation
    with torch.no_grad():
        for _, inputs, answers in tqdm(val_loader, desc="Validation"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=16,
                truncation=False,
            ).input_ids.to(device)

            # Create attention mask for padding tokens
            attention_mask = labels != processor.tokenizer.pad_token_id

            # Move inputs to device and cast pixel_values to bf16
            inputs = inputs.to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, cache_enabled=True):
                outputs = safe_forward(
                    model=model,
                    inputs=inputs,
                    labels=labels,
                    attention_mask=attention_mask,
                    max_chunks=config["eval_batch_size"]
                )
                total_loss += outputs.loss.item()
                del inputs, answers, labels, attention_mask, outputs

            steps += 1

        # Sample predictions
        for tasks, inputs, answers in val_loader:
            # Move inputs to device and cast pixel_values to bf16
            inputs = inputs.to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

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

            del generated_ids

            for prompt, gen_text, answer in zip(tasks, generated_texts, answers):
                parsed = processor.post_process_generation(
                    gen_text,
                    task=prompt,
                    image_size=(
                        inputs["pixel_values"].shape[-2],
                        inputs["pixel_values"].shape[-1],
                    ),
                )
                # Log to wandb table
                table.add_data(answer, parsed[prompt].replace("<pad>", ""))
                print("\n----")
                print("      Prompt:", prompt)
                print("Ground Truth:", answer)
                print("  Prediction:", parsed[prompt].replace("<pad>", ""))
                print("----")

                del parsed, prompt, gen_text, answer

            del tasks, inputs, answers

            break  # First batch only

    run.log({"validation/avg_loss": total_loss / steps, "validation/predictions": table}, step=current_step)

    del table, total_loss


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


def main():
    parser = argparse.ArgumentParser(
        description="A simple Florence-2 finetuning script."
    )

    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the yaml training config file."
    )

    args = parser.parse_args()

    with open(args.yaml_file, "r") as f:
        config = yaml.safe_load(f)

    # TODO: Set default config values if setting is not present

    processor = AutoProcessor.from_pretrained(config["model_name"], trust_remote_code=True)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # Gather and filter data
    print("Gathering files from multiple datasets...")
    all_pairs = get_all_files_by_prompt(config["dataset_config"])

    gc.collect()

    processes_count = int(
        (config["dataloader_workers"] or os.cpu_count()) * config["filtering_processes_per_thread"]
    )

    print(
        f"Filtering data based on token length using {processes_count} processes "
        f"and batch size {int(config['filtering_batch_size'])}"
    )
    filtered_pairs = filter_all_pairs(all_pairs, processor, processes_count, int(config["filtering_batch_size"]))
    print(f"Filtered out {len(all_pairs) - len(filtered_pairs)} files due to token length.")

    del all_pairs

    random.shuffle(filtered_pairs)
    eval_size = (
        int(len(filtered_pairs) * config["eval_split"]) if config["eval_split"] < 1
        else min(int(config["eval_split"]), len(filtered_pairs))
    )
    eval_dataset_pairs = filtered_pairs[:eval_size]
    train_dataset_pairs = filtered_pairs[eval_size:]

    del filtered_pairs

    # Prepare output directory
    output_base_dir = Path(f"./checkpoints/{config['run_name']}")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Save YAML file to output dir
    shutil.copy(args.yaml_file, output_base_dir / Path(args.yaml_file).name)

    # Save eval image paths
    eval_list_file = output_base_dir / "eval_image_paths.txt"
    with open(eval_list_file, "w") as f:
        for _, img_path, _ in eval_dataset_pairs:
            f.write(f"{img_path}\n")
    print(f"Saved evaluation image paths to {eval_list_file}")

    del eval_list_file

    train_dataset = LocalImageTextDataset(train_dataset_pairs)
    # TODO: Add ability to specify a val set
    val_dataset = LocalImageTextDataset(eval_dataset_pairs)

    del train_dataset_pairs, eval_dataset_pairs

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train_batch_size"]),
        collate_fn=partial(collate_fn, processor=processor),
        shuffle=True,
        num_workers=int(config["dataloader_workers"]) or os.cpu_count(),
        persistent_workers=config["persistent_workers"],
        pin_memory=True,
        prefetch_factor=int(config["dataloader_prefetch_factor"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["eval_batch_size"]),
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=int(config["dataloader_workers"]) or os.cpu_count(),
        persistent_workers=config["persistent_workers"],
        pin_memory=True,
        prefetch_factor=int(config["dataloader_prefetch_factor"]),
    )

    del train_dataset, val_dataset

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=config["attn_implementation"],
    )

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
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    print(f"Moving model to {device}")
    model.to(device)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Train the model
    print("Starting training")
    train_model(train_loader, val_loader, model, processor, config)


if __name__ == "__main__":
    main()
