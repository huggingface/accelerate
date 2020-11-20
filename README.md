# accelerate

## Installation

Install PyTorch, then

```bash
git clone https://github.com/huggingface/accelerate.git
cd accelerate
pip install -e .
```

## Tests

To run the example script on multi-GPU:
```bash
python -m torch.distributed.launch --nproc_per_node 2 examples/glue_example.py
```

To run the example script on TPUs:
```bash
python tests/xla_spawn.py --num_cores 8 examples/glue_example.py
```