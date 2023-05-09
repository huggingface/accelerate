# What are these scripts?

All scripts in this folder originate from the `nlp_example.py` file, as it is a very simplistic NLP training example using Accelerate with zero extra features.

From there, each further script adds in just **one** feature of Accelerate, showing how you can quickly modify your own scripts to implement these capabilities.

A full example with all of these parts integrated together can be found in the `complete_nlp_example.py` script and `complete_cv_example.py` script.

Adjustments to each script from the base `nlp_example.py` file can be found quickly by searching for "# New Code #"

## Example Scripts by Feature and their Arguments

### Base Example (`../nlp_example.py`)

- Shows how to use `Accelerator` in an extremely simplistic PyTorch training loop
- Arguments available:
  - `mixed_precision`, whether to use mixed precision. ("no", "fp16", or "bf16")
  - `cpu`, whether to train using only the CPU. (yes/no/1/0)

All following scripts also accept these arguments in addition to their added ones.

These arguments should be added at the end of any method for starting the python script (such as `python`, `accelerate launch`, `python -m torch.distributed.run`), such as:

```bash
accelerate launch ../nlp_example.py --mixed_precision fp16 --cpu 0
```

### Checkpointing and Resuming Training (`checkpointing.py`)

- Shows how to use `Accelerator.save_state` and `Accelerator.load_state` to save or continue training
- **It is assumed you are continuing off the same training script**
- Arguments available:
  - `checkpointing_steps`, after how many steps the various states should be saved. ("epoch", 1, 2, ...)
  - `output_dir`, where saved state folders should be saved to, default is current working directory
  - `resume_from_checkpoint`, what checkpoint folder to resume from. ("epoch_0", "step_22", ...)

These arguments should be added at the end of any method for starting the python script (such as `python`, `accelerate launch`, `python -m torchrun`), such as:

(Note, `resume_from_checkpoint` assumes that we've ran the script for one epoch with the `--checkpointing_steps epoch` flag)

```bash
accelerate launch ./checkpointing.py --checkpointing_steps epoch output_dir "checkpointing_tutorial" --resume_from_checkpoint "checkpointing_tutorial/epoch_0"
```

### Cross Validation (`cross_validation.py`)

- Shows how to use `Accelerator.free_memory` and run cross validation efficiently with `datasets`.
- Arguments available:
  - `num_folds`, the number of folds the training dataset should be split into.

These arguments should be added at the end of any method for starting the python script (such as `python`, `accelerate launch`, `python -m torchrun`), such as:

```bash
accelerate launch ./cross_validation.py --num_folds 2
```

### Experiment Tracking (`tracking.py`)

- Shows how to use `Accelerate.init_trackers` and `Accelerator.log`
- Can be used with Weights and Biases, TensorBoard, or CometML.
- Arguments available:
  - `with_tracking`, whether to load in all available experiment trackers from the environment.

These arguments should be added at the end of any method for starting the python script (such as `python`, `accelerate launch`, `python -m torchrun`), such as:

```bash
accelerate launch ./tracking.py --with_tracking
```

### Gradient Accumulation (`gradient_accumulation.py`)

- Shows how to use `Accelerator.no_sync` to prevent gradient averaging in a distributed setup.
- Arguments available:
  - `gradient_accumulation_steps`, the number of steps to perform before the gradients are accumulated and the optimizer and scheduler are stepped + zero_grad

These arguments should be added at the end of any method for starting the python script (such as `python`, `accelerate launch`, `python -m torchrun`), such as:

```bash
accelerate launch ./gradient_accumulation.py --gradient_accumulation_steps 5
```

### LocalSGD (`local_sgd.py`)
- Shows how to use `Accelerator.no_sync` to prevent gradient averaging in a distributed setup. However, unlike gradient accumulation, this method does not change the effective batch size. Local SGD can be combined with gradient accumulation.

These arguments should be added at the end of any method for starting the python script (such as `python`, `accelerate launch`, `python -m torchrun`), such as:

```bash
accelerate launch ./local_sgd.py --local_sgd_steps 4
```


