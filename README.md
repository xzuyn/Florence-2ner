# Florence-2ner

This script provides a straightforward way to fine-tune [Microsoft's Florence-2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de) models for image captioning tasks. It's designed to be configurable and integrates with [Weights & Biases](https://wandb.ai/site) for experiment tracking.

## Key Features

* **Easy Data Loading:** Handles local image and text file pairs for training in a similar structure to [kohya_ss](https://github.com/bmaltais/kohya_ss).
* **Gradient Accumulation:** Supports gradient accumulation to train with larger effective batch sizes on limited GPU memory.
* **Gradient Checkpointing:**  Optionally enables gradient checkpointing to reduce memory usage during training.
* **Parameter Freezing:** Allows freezing specific parts of the model (vision tower, language model, or other components) for targeted fine-tuning.
* **Learning Rate Scheduling:** Implements a linear warmup followed by cosine annealing with warm restarts for optimized learning.
* **Evaluation during Training:** Evaluates the model on a validation set at specified intervals.
* **Checkpoint Saving:** Saves model checkpoints periodically, with options to limit the number of saved checkpoints.
* **Weights & Biases Integration:** Seamlessly logs training metrics, validation losses, and generated captions to wandb.

## Instructions

This script requires a folder containing images and corresponding caption files in plain text format.

```
dataset_folder
  └── train  # Optional subdirectory
      ├── image1.jpg
      ├── image1.txt
      ├── another image.webp
      ├── another image.txt
      ├── example file name.png
      └── example file name.txt
```

## Configuration

The training process is customizable through the `config` dictionary at the beginning of the script:

```python
config = {
    "model_name": "microsoft/Florence-2-base",  # The pre-trained Florence-2 model to use
    "dataset_path": "",                         # Path to your dataset folder
    "run_name": "",                             # Name for your training run (used for wandb and checkpoints)
    "epochs": 2,                                # Number of training epochs
    "learning_rate": 1e-5,                      # Initial learning rate
    "gradient_checkpointing": True,             # Enable gradient checkpointing to save memory
    "freeze_vision": True,                      # Freeze the vision encoder
    "freeze_language": False,                   # Freeze the language model
    "freeze_other": False,                      # Freeze the rest
    "train_batch_size": 16,                     # Batch size for training
    "eval_batch_size": 16,                      # Batch size for evaluation
    "gradient_accumulation_steps": 16,          # Number of steps for gradient accumulation
    "clip_grad_norm": 1,                        # Maximum norm for gradient clipping
    "weight_decay": 1e-5,                       # Weight decay for the optimizer
    "save_steps": 50,                           # Save a checkpoint every this many steps
    "save_total_limit": 3,                      # Keep only the last this many checkpoints
    "eval_steps": 50,                           # Evaluate the model every this many steps
    "warmup_steps": 50,                         # Number of warmup steps for the learning rate scheduler
    "eval_split_ratio": 0.1,                    # Ratio of the dataset to use for validation
    "seed": 42                                  # Random seed for reproducibility
}
```

## Checkpoints

The script saves model checkpoints in the `./checkpoints/<run_name>/` directory. Each checkpoint is saved in a subdirectory named `step-<current_step>`. The `save_total_limit` parameter controls the number of recent checkpoints to keep, automatically deleting older ones.

## Notes

This README was written mainly by `gemini-2.0-flash-thinking-exp-1219`.
