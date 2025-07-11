import os
import gc
import re
import math
import wandb
import json
import yaml
import uuid
import torch
import shutil
import psutil
import random
import logging
import argparse
import evaluate
import multiprocessing

from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from functools import partial


# Allow extremely large images
Image.MAX_IMAGE_PIXELS = None

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ACTIVATION_OFFLOAD_DIR = "./TEMP_ACTIVATIONS"


def logger_setup(run_name, file_location):
    global logger

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{file_location}/{run_name}.log"),
            logging.StreamHandler()
        ],
        force=True,
    )
    logger = logging.getLogger()
    logger.info(run_name)


# https://github.com/IvanVassi/REX_LR
class RexLR(torch.optim.lr_scheduler.LRScheduler):
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

    def __init__(self, optimizer, max_lr, min_lr, total_steps=0, num_warmup_steps=0, last_step=0):
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


class LocalImageTextDataset(torch.utils.data.Dataset):
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
            logger.exception(f"Error loading data from {image_path}: {e}")
            raise


class SelfDeletingTempFile:
    def __init__(self):
        self.name = os.path.join(ACTIVATION_OFFLOAD_DIR, str(uuid.uuid4()))

    def __del__(self):
        os.remove(self.name)


def gpu_pack(tensor):
    return tensor


def gpu_unpack(tensor):
    return tensor


def cpu_pack(tensor):
    return tensor.cpu()  # TODO: Add a "partial offload" option


def cpu_unpack(tensor):
    return tensor.to(device)


def disk_pack(tensor):
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file


def disk_unpack(temp_file):
    return torch.load(temp_file.name, map_location=device, weights_only=True)


@torch.inference_mode()
def run_generate(model, input_ids, pixel_values, do_gc):
    if do_gc:
        logger.info("Running garbage collection")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    return model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        num_beams=5,
        min_p=0.05,
        do_sample=True,
    )


@torch.inference_mode()
def run_forward(model, input_ids, pixel_values, labels, attention_mask, do_gc):
    if do_gc:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    # https://unsloth.ai/blog/gradient
    batch_token_count = attention_mask.sum()
    loss_sum = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        labels=labels,
        attention_mask=attention_mask,
    ).loss.mul_(batch_token_count)

    return loss_sum.detach().item(), batch_token_count.item()


def run_forward_backward(model, input_ids, pixel_values, labels, attention_mask, do_gc, activation_offloading):
    if do_gc:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    if activation_offloading == "cpu":
        pack_choice, unpack_choice = cpu_pack, cpu_unpack
    elif activation_offloading == "disk":
        pack_choice, unpack_choice = disk_pack, disk_unpack
    else:
        pack_choice, unpack_choice = gpu_pack, gpu_unpack

    with torch.autograd.graph.saved_tensors_hooks(pack_choice, unpack_choice):
        # https://unsloth.ai/blog/gradient
        batch_token_count = attention_mask.sum()
        loss_sum = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
            attention_mask=attention_mask,
        ).loss.mul_(batch_token_count)

    loss_sum.backward()

    return loss_sum.detach().item(), batch_token_count.item()


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

        logger.info(f"{task_prompt}: found {count} pairs")

    return all_pairs


def filter_data_chunk(chunk, processor):
    texts = []
    meta = []
    for task_prompt, img_path, txt_path in chunk:
        try:
            with open(txt_path, "r") as f:
                text = f.read().replace("  ", " ").strip()
            texts.append(text)
            meta.append((task_prompt, img_path, txt_path))
        except Exception as e:
            logger.exception(f"Error reading {txt_path}: {e}")

    if not texts:
        return []

    tokenized = processor.tokenizer(
        texts,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    filtered = []
    for (task_prompt, img_path, txt_path), ids in zip(meta, tokenized.input_ids):
        length = len(ids)
        if length <= (processor.tokenizer.model_max_length - 24):  # TODO: Properly calculate (1024 - task prompt token count)
            filtered.append((task_prompt, img_path, txt_path))
        else:
            logger.warning(f"Caption too long ({length} tokens): {img_path}")

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
        logger.exception(f"Processor error: {e}")
        raise

    return list(tasks), inputs, list(answers)


# TODO: Add more optimizers
# https://github.com/warner-benjamin/optimi
# https://github.com/yangluo7/CAME
def prepare_optimizer(model_parameters, optimizer_name, optimizer_lr, optimizer_weight_decay):
    logger.info("Preparing optimizer")

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
            logger.exception("You do not have optimī installed. Please install it using `pip install torch-optimi`")
            raise
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
            logger.exception(
                "You do not have CAME installed. "
                "Please install it using `pip install git+https://github.com/xzuyn/CAME.git@sr-grams-cautious-8bit`"
            )
            raise
    else:
        logger.error("No valid optimizer selected. Options are: `OptimiAdamW` & `CAME`.")
        raise

    return optimizer


# TODO: Add more LR schedulers
def prepare_lr_scheduler(
    scheduler_optimizer,
    scheduler_name,
    scheduler_lr,
    scheduler_min_lr,
    scheduler_warmup_steps,
    scheduler_total_training_steps
):
    logger.info("Preparing lr_scheduler")

    if scheduler_name == "Constant":
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(
            scheduler_optimizer
        )
    elif scheduler_name == "Cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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
        logger.warning("No valid LR scheduler selected. Falling back to Cosine.")
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            scheduler_optimizer,
            T_max=(scheduler_total_training_steps - scheduler_warmup_steps) + 1
        )

    if scheduler_warmup_steps > 0 and scheduler_name != "REX":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            scheduler_optimizer,
            start_factor=1e-20,
            end_factor=1.0,
            total_iters=scheduler_warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            scheduler_optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[scheduler_warmup_steps]
        )
    else:
        scheduler = main_scheduler

    return scheduler


def save_model_checkpoint(model, processor, run_name, train_steps, save_total_limit):
    logger.info("Saving checkpoint")

    if config.get("debug"):
        logger.info("Running garbage collection")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    output_dir = f"./checkpoints/{run_name}/step-{train_steps}"
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
            logger.info(f"Deleting old checkpoint: {checkpoint_to_delete}")
            shutil.rmtree(checkpoint_to_delete)


def train_model(model, model_dtype, optimizer, scheduler, train_loader, val_loader, processor, config, run):
    logger.info("Starting training")

    total_training_steps = (
        math.ceil(len(train_loader) / config.get("gradient_accumulation_steps")) * config.get("epochs")
    )
    train_steps = 0
    window_loss_sum = 0.0
    window_token_count = 0

    # Evaluate before any training starts
    if config.get("eval_before_training"):
        evaluate_model(model, model_dtype, val_loader, processor, config, run, train_steps)

    logger.info("Setting model to train mode")
    model.train()

    optimizer.zero_grad()

    progress_bar = tqdm(range(total_training_steps), desc="Training")
    for epoch in range(config.get("epochs")):
        for i, (_, inputs, answers) in enumerate(train_loader):
            if config.get("debug"):
                logger.info("DEBUG - Creating labels")
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                pad_to_multiple_of=16,
                return_token_type_ids=False,
            ).input_ids.to(device)

            # Move inputs to device and cast pixel_values to bf16 or fp16
            if config.get("debug"):
                logger.info("DEBUG - Moving inputs")
            inputs = inputs.to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

            if config.get("debug"):
                logger.info("DEBUG - Running forward & backward")
            loss_sum, token_count = run_forward_backward(
                model=model,
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                labels=labels,
                attention_mask=(labels != processor.tokenizer.pad_token_id),  # type: ignore[attr-defined]
                do_gc=config.get("garbage_collection"),
                activation_offloading=config.get("activation_offloading"),
            )
            window_loss_sum += loss_sum
            window_token_count += token_count

            del inputs, answers, labels

            if (i + 1) % config.get("gradient_accumulation_steps") == 0 or (i + 1) == len(train_loader):
                if config.get("debug"):
                    logger.info("DEBUG - Clipping grad_norm")
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(window_token_count)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("clip_grad_norm"))

                if config.get("debug"):
                    logger.info("DEBUG - Stepping optimizer & scheduler")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_steps += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "epoch": train_steps / (total_training_steps / config.get("epochs")),
                        "loss": window_loss_sum / window_token_count,
                        "grad_norm": grad_norm.item(),
                    }
                )

                run.log(
                    {
                        "train/loss": window_loss_sum / window_token_count,
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": train_steps / (total_training_steps / config.get("epochs")),
                    }
                )

                window_loss_sum = 0.0
                window_token_count = 0

                if train_steps % config.get("save_steps") == 0:
                    save_model_checkpoint(
                        model,
                        processor,
                        config.get("run_name"),
                        train_steps,
                        config.get("save_total_limit"),
                    )

                if train_steps % config.get("eval_steps") == 0:
                    evaluate_model(model, model_dtype, val_loader, processor, config, run, train_steps)
                    logger.info("Setting model to train mode")
                    model.train()

    # Save the last step if it hasn't been already
    if train_steps % config.get("save_steps") != 0:
        save_model_checkpoint(
            model,
            processor,
            config.get("run_name"),
            train_steps,
            config.get("save_total_limit"),
        )

    # Eval the last step if it hasn't been already
    if train_steps % config.get("eval_steps") != 0:
        evaluate_model(model, model_dtype, val_loader, processor, config, run, train_steps)
        logger.info("Setting model to train mode")
        model.train()


@torch.inference_mode()
def evaluate_model(model, model_dtype, val_loader, processor, config, run, train_steps):
    logger.info("Starting evaluation")

    if config.get("debug"):
        logger.info("Running garbage collection")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    all_references = []
    all_predictions = []
    eval_window_loss_sum = 0.0
    eval_window_token_count = 0

    logger.info("Setting model to eval mode")
    model.eval()

    eval_progress_bar = tqdm(range(len(val_loader)), desc="Validating")
    for val_idx, (tasks, inputs, answers) in enumerate(val_loader):
        if config.get("debug"):
            logger.info("DEBUG - Creating labels")
        labels = processor.tokenizer(
            text=answers,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=16,
            truncation=False,
        ).input_ids.to(device)

        # Move inputs to device and cast pixel_values to bf16 or fp16
        if config.get("debug"):
            logger.info("DEBUG - Moving inputs")
        inputs = inputs.to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

        if config.get("debug"):
            logger.info("DEBUG - Running forward")
        eval_loss_sum, eval_token_count = run_forward(
            model=model,
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            labels=labels,
            attention_mask=(labels != processor.tokenizer.pad_token_id),  # type: ignore[attr-defined]
            do_gc=config.get("do_gc"),
        )
        eval_window_loss_sum += eval_loss_sum
        eval_window_token_count += eval_token_count

        if config.get("do_extra_eval") or (config.get("print_first_batch_predictions") and val_idx == 0):
            if config.get("debug"):
                logger.info("DEBUG - Running generate")
            generated_ids = run_generate(
                model=model,
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                do_gc=config.get("do_gc"),
            )

            if config.get("debug"):
                logger.info("DEBUG - Decoding predictions")
            predictions = [
                gen_text.strip() for gen_text in processor.batch_decode(generated_ids, skip_special_tokens=True)
            ]

            if config.get("do_extra_eval"):
                if config.get("debug"):
                    logger.info("DEBUG - Extending prediction & reference lists")
                all_predictions.extend(predictions)
                all_references.extend(answers)

            # Print the predictions of the first batch only
            if config.get("print_first_batch_predictions") and val_idx == 0:
                if config.get("debug"):
                    logger.info("DEBUG - Printing first batch")
                for task, prediction, answer in zip(tasks, predictions, answers):
                    print("\n----")
                    print("        Task:", task)
                    print("Ground Truth:", answer)
                    print("  Prediction:", prediction)
                    print("----")

        eval_progress_bar.update(1)
        eval_progress_bar.set_postfix({"loss": eval_loss_sum / eval_token_count})

    wandb_data = {"validation/avg_loss": eval_window_loss_sum / eval_window_token_count}

    if config.get("do_extra_eval"):
        logger.info("Loading extra evaluation metrics")
        google_bleu_metric = evaluate.load("google_bleu")
        meteor_metric = evaluate.load("meteor")
        rouge_metric = evaluate.load("rouge")

        logger.info("Running google_bleu eval")
        avg_google_bleu = google_bleu_metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        logger.info("Running meteor eval")
        avg_meteor = meteor_metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        logger.info("Running rouge eval")
        avg_rouge = rouge_metric.compute(
            predictions=all_predictions,
            references=all_references,
            rouge_types=["rouge1", "rouge2", "rougeL"]
        )

        wandb_data.update(
            {
                "validation/avg_google_bleu": avg_google_bleu["google_bleu"],
                "validation/avg_meteor": avg_meteor["meteor"],
                "validation/avg_rouge1": avg_rouge["rouge1"],
                "validation/avg_rouge2": avg_rouge["rouge2"],
                "validation/avg_rougeL": avg_rouge["rougeL"],
            }
        )

        del google_bleu_metric, meteor_metric, rouge_metric, all_references, all_predictions

    logger.info("Logging evaluation metrics to W&B")
    run.log(wandb_data, step=train_steps)

    if config.get("debug"):
        logger.info("Running garbage collection")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


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
        # TODO: Set default config values if setting is not present
        config = yaml.safe_load(f)

    # Prepare output directory
    output_base_dir = Path(f"./checkpoints/{config['run_name']}")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logger_setup(config.get("run_name"), output_base_dir)
    logger.info(str(config))
    if config.get("debug"):
        logger.info("DEBUG - Debug mode enabled")

    if config.get("activation_offloading") == "disk":
        os.makedirs(ACTIVATION_OFFLOAD_DIR, exist_ok=True)

    model_dtype = torch.bfloat16 if config.get("use_bf16") else torch.float16
    processor = AutoProcessor.from_pretrained(config.get("model_name"), trust_remote_code=True, padding_side="left")

    random.seed(config.get("seed"))
    torch.manual_seed(config.get("seed"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed"))

    # Gather and filter data
    logger.info("Gathering files from multiple datasets...")
    all_pairs = get_all_files_by_prompt(config.get("dataset_config"))

    processes_count = int(
        (config.get("dataloader_workers") or os.cpu_count()) * config.get("filtering_processes_per_thread")
    )

    logger.info(
        f"Filtering data based on token length using {processes_count} processes "
        f"and batch size {int(config['filtering_batch_size'])}"
    )
    filtered_pairs = filter_all_pairs(all_pairs, processor, processes_count, int(config.get("filtering_batch_size")))
    logger.info(f"Filtered out {len(all_pairs) - len(filtered_pairs)} files due to token length.")

    del all_pairs

    logger.info("Shuffling and splitting train/eval splits")
    random.shuffle(filtered_pairs)
    eval_size = (
        int(len(filtered_pairs) * config.get("eval_split")) if config.get("eval_split") < 1
        else min(int(config.get("eval_split")), len(filtered_pairs))
    )
    eval_dataset_pairs = filtered_pairs[:eval_size]
    train_dataset_pairs = filtered_pairs[eval_size:]

    del filtered_pairs

    # Save YAML file to output dir
    shutil.copy(args.yaml_file, output_base_dir / Path(args.yaml_file).name)

    # Save eval image paths
    eval_list_file = output_base_dir / "eval_image_paths.txt"
    logger.info(f"Saving evaluation image paths to {eval_list_file}")
    with open(eval_list_file, "w") as f:
        for _, img_path, _ in eval_dataset_pairs:
            f.write(f"{img_path}\n")

    del eval_list_file

    train_dataset = LocalImageTextDataset(train_dataset_pairs)
    # TODO: Add ability to specify an eval set
    # TODO: Make eval optional
    val_dataset = LocalImageTextDataset(eval_dataset_pairs)

    del train_dataset_pairs, eval_dataset_pairs

    logger.info("Loading dataloaders")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config.get("train_batch_size")),
        collate_fn=partial(collate_fn, processor=processor),
        shuffle=True,
        num_workers=int(config.get("dataloader_workers")) or os.cpu_count(),
        persistent_workers=config.get("persistent_workers"),
        pin_memory=True,
        prefetch_factor=int(config.get("dataloader_prefetch_factor")),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(config.get("eval_batch_size")),
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=int(config.get("dataloader_workers")) or os.cpu_count(),
        persistent_workers=config.get("persistent_workers"),
        pin_memory=True,
        prefetch_factor=int(config.get("dataloader_prefetch_factor")),
    )

    del train_dataset, val_dataset

    logger.info("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        config.get("model_name"),
        torch_dtype=model_dtype,
        trust_remote_code=True,
        attn_implementation=config.get("attn_implementation"),
    )

    if config.get("freeze_language"):
        for param in model.language_model.parameters():
            param.requires_grad = False
    if config.get("freeze_vision"):
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    if config.get("freeze_other"):
        model.image_pos_embed.column_embeddings.weight.requires_grad = False
        model.image_pos_embed.row_embeddings.weight.requires_grad = False
        model.visual_temporal_embed.pos_idx_to_embed.requires_grad = False
        model.image_proj_norm.bias.requires_grad = False
        model.image_proj_norm.weight.requires_grad = False
        model.image_projection.requires_grad = False

    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    logger.info(f"Moving model to {device}")
    model.to(device)

    optimizer = prepare_optimizer(
        model.parameters(),
        config.get("optimizer"),
        config.get("learning_rate"),
        config.get("weight_decay")
    )

    scheduler = prepare_lr_scheduler(
        optimizer,
        config.get("lr_scheduler"),
        config.get("learning_rate"),
        config.get("min_learning_rate") or 0,
        config.get("warmup_steps"),
        math.ceil(len(train_loader) / config.get("gradient_accumulation_steps")) * config.get("epochs"),
    )

    # Train the model
    with wandb.init(
        project=config.get("wandb_project_name"),
        name=config.get("run_name"),
        save_code=True,
        settings=wandb.Settings(x_stats_sampling_interval=config.get("wandb_x_stats_sampling_interval")),
    ) as run:
        wandb.save(output_base_dir / Path(args.yaml_file).name)

        train_model(model, model_dtype, optimizer, scheduler, train_loader, val_loader, processor, config, run)


if __name__ == "__main__":
    main()
