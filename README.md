<p align="center">
    <br>
    <img src="docs/source/imgs/accelerate_logo.png" width="400"/>
    <br>
<p>

<p align="center">
    <!-- Uncomment when CircleCI is setup
    <a href="https://circleci.com/gh/huggingface/accelerate">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    -->
    <a href="https://github.com/huggingface/accelerate/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/accelerate.svg?color=blue">
    </a>
    <!-- Uncomment when doc is online
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    -->
    <a href="https://github.com/huggingface/accelerate/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/accelerate.svg">
    </a>
    <a href="https://github.com/huggingface/accelerate/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

<h3 align="center">
<p>Run your *raw* PyTorch training script on any kind of device
</h3>

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