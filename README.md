# accelerate

## Installation

Install PyTorch, then

```bash
git clone https://github.com/huggingface/accelerate.git
cd accelerate
pip install -e .
```

## Tests

### Using the accelerate CLI

Create a default config for your environment with
```bash
accelerate config
```
then launch the GLUE example with
```bash
accelerate launch examples/glue_example.py --task_name mrpc --model_name_or_path bert-base-cased
```

### Traditional launchers

To run the example script on multi-GPU:
```bash
python -m torch.distributed.launch --nproc_per_node 2 --use_env examples/glue_example.py \
    --task_name mrpc --model_name_or_path bert-base-cased
```

To run the example script on TPUs:
```bash
python tests/xla_spawn.py --num_cores 8 examples/glue_example.py\
    --task_name mrpc --model_name_or_path bert-base-cased
```