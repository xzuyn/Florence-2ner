# Florence-2ner

This script provides a straightforward way to fine-tune [Microsoft's Florence-2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de) models for image captioning tasks.

## Usage

`python train.py example.yaml`

## Dataset Format

Same as what you'd use for kohya_ss or OneTrainer. It is recursive, so pairs within subfolders of the folder you select will be seen too.

```
image_folder
└── sub_folder
    └── bleaasdasfads.jpg
    └── bleaasdasfads.txt
└── sub_folder2
    └── fghjtryjt.jpg
    └── fghjtryjt.txt
└── image_file1.jpg
└── image_file1.txt
└── blah blah.jpg
└── blah blah.txt
└── asd87sdf78sdfg7.jpg
└── asd87sdf78sdfg7.txt
```

The YAML config maps task prompts to lists of folders containing this structure.

## Usage

```bash
python train_florence2.py config.yaml
```

## Example YAML Config

```yaml
# Project Settings
wandb_project_name: Florence-2-base
run_name: Florence-2-base-Example-v0.1-run1
wandb_x_stats_sampling_interval: 1

# Model Settings
model_name: microsoft/Florence-2-base
attn_implementation: sdpa
use_bf16: true

# Save & Eval Settings
save_steps: 10
eval_steps: 10
save_total_limit: 3
do_extra_eval: true
eval_before_training: true
print_first_batch_predictions: true

# Hyperparameter Settings
epochs: 2
optimizer: CAME
lr_scheduler: REX
learning_rate: 0.0000025
min_learning_rate: 0.00000025
train_batch_size: 4
eval_batch_size: 4
gradient_accumulation_steps: 16
gradient_checkpointing: true
warmup_steps: 20
clip_grad_norm: 0.5
weight_decay: 0.01
seed: 42
garbage_collection: false
activation_offloading:  # cpu or disk or partial_cpu or partial_disk

# Freezing Settings
freeze_vision: false
freeze_language: false
freeze_other: false

# Dataset Settings
eval_split: 256
dataset_config:
  <CAPTION>:
    - "/media/xzuyn/NVMe/Datasets/Images/Example-Short"
  <DETAILED_CAPTION>:
    - "/media/xzuyn/NVMe/Datasets/Images/Example-Long"

# Filtering Settings
filtering_processes_per_thread: 1
filtering_batch_size: 16

# Dataloader Settings
dataloader_workers: 12
persistent_workers: true
dataloader_prefetch_factor: 2

# Debug
debug: false
```
