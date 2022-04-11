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

### Checkpointing and Resuming Training (`checkpointing.py`)

- Shows how to use `Accelerator.save_state` and `Accelerator.load_state` to save or continue training
- **It is assumed you are continuing off the same training script**
- Arguments available:
  - `checkpointing_steps`, after how many steps the various states should be saved. ("epoch", 1, 2, ...)
  - `output_dir`, where saved state folders should be saved to, default is current working directory
  - `resume_from_checkpoint`, what checkpoint folder to resume from. ("epoch_0", "step_22", ...)

### Experiment Tracking (`tracking.py`)

- Shows how to use `Accelerate.init_trackers` and `Accelerator.log`
- Can be used with Weights and Biases, TensorBoard, or CometML.
- Arguments available:
  - `with_tracking`, whether to load in all available experiment trackers from the environment.