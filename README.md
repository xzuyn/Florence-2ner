# Florence-2ner

This script provides a straightforward way to fine-tune [Microsoft's Florence-2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de) models for image captioning tasks.

## Usage

`python train.py example.yaml`

## Dataset Format

Each folder should contain the images, as well as subfolders of captions named using the task prompts. This way you can train multiple tasks on an image, while only needing to have a single copy of the image.

```
Example1
└── <CAPTION>
    └── image_file1.txt
    └── bleaasdasfads.txt
    └── fghjtryjt.txt
    └── blah blah.txt
    └── asd87sdf78sdfg7.txt
└── <DETAILED_CAPTION>
    └── image_file1.txt
    └── bleaasdasfads.txt
    └── fghjtryjt.txt
    └── blah blah.txt
    └── asd87sdf78sdfg7.txt
└── image_file1.jpg
└── bleaasdasfads.webp
└── blah blah.png
└── asd87sdf78sdfg7.jpeg
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
wandb_x_stats_sampling_interval: 10  # used for logging system stats every x seconds, default is 10

# Model Settings
model_name: microsoft/Florence-2-base
attn_implementation: sdpa  # sdpa, eager, or flash_attention_2
use_bf16: true  # true or false, will use fp16 if false

# Save & Eval Settings
save_steps: 10
eval_steps: 10
save_total_limit: 3
do_extra_eval: true  # true or false
eval_before_training: true  # true or false
print_first_batch_predictions: true  # true or false

# Hyperparameter Settings
epochs: 2
optimizer: CAME
lr_scheduler: REX
learning_rate: 0.0000025
min_learning_rate: 0.00000025
train_batch_size: 4
eval_batch_size: 4
gradient_accumulation_steps: 16
gradient_checkpointing: # true or false, may not even work
warmup_steps: 20
clip_grad_norm: 0.5
weight_decay: 0.01
seed: 42
garbage_collection: # true or false

# Offloading Settings
activation_offloading:  # cpu or disk or hybrid
# used with hybrid activation offloading. max amount that can be kept on gpu (default is 10240 which is 10GB)
gpu_limit_mb:  
# used with hybrid activation offloading. max amount that can be moved to cpu (default is 10240 which is 10GB)
offload_cpu_limit_mb:

# Freezing Settings
freeze_vision: # true or false
freeze_language: # true or false
freeze_other: # true or false

# Dataset Settings
eval_split: 256
dataset_config:
  - "/media/xzuyn/NVMe/Datasets/Images/Example1"
  - "/media/xzuyn/NVMe/Datasets/Images/Example2"

# Filtering Settings
filtering_processes_per_thread: 1
filtering_batch_size: 16

# Dataloader Settings
dataloader_workers: 12
persistent_workers: true  # true or false
dataloader_prefetch_factor: 2
```
