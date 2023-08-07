<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Quantization

## `bitsandbytes` Integration

🤗 Accelerate brings `bitsandbytes` quantization to your model. You can now load any pytorch model in 8-bit or 4-bit with a few lines of code.

If you want to use 🤗 Transformers models with `bitsandbytes`, you should follow this [documentation](https://huggingface.co/docs/transformers/main_classes/quantization). 

To learn more about how the `bitsandbytes` quantization works, check out the blog posts on [8-bit quantization](https://huggingface.co/blog/hf-bitsandbytes-integration) and [4-bit quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

### Pre-Requisites
You will need to install the following requirements:

- Install `bitsandbytes` library
```bash
pip install bitsandbytes
```
- Install latest `accelerate` from source
```bash
pip install git+https://github.com/huggingface/accelerate.git
```
- Install `minGPT` and `huggingface_hub` to run examples
```bash
git clone https://github.com/karpathy/minGPT.git
pip install minGPT/
pip install huggingface_hub
```

### How it works

First, we need to initialize our model. To save memory, we can initialize an empty model using the context manager [`init_empty_weights`]. 

Let's take the GPT2 model from minGPT library.
```py
from accelerate import init_empty_weights
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = 50257
model_config.block_size = 1024

with init_empty_weights():
    empty_model = GPT(model_config)
```

Then, we need to get the path to the weights of your model. The path can be the state_dict file (e.g. "pytorch_model.bin") or a folder containing the sharded checkpoints. 

```py
from huggingface_hub import snapshot_download
weights_location = snapshot_download(repo_id="marcsun13/gpt2-xl-linear-sharded")
```

Finally, you need to set your quantization configuration with [`~utils.BnbQuantizationConfig`].

Here's an example for 8-bit quantization:
```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
```

Here's an example for 4-bit quantization:
```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
```

To quantize your empty model with the selected configuration, you need to use [`~utils.load_and_quantize_model`]. 

```py
from accelerate.utils import load_and_quantize_model
quantized_model = load_and_quantize_model(empty_model, weights_location=weights_location, bnb_quantization_config=bnb_quantization_config, device_map = "auto")
```

### Saving and loading 8-bit model

You can save your 8-bit model with accelerate using [`~Accelerator.save_model`]. 

```py
from accelerate import Accelerator
accelerate = Accelerator()
new_weights_location = "path/to/save_directory"
accelerate.save_model(quantized_model, new_weights_location)

quantized_model_from_saved = load_and_quantize_model(empty_model, weights_location=new_weights_location, bnb_quantization_config=bnb_quantization_config, device_map = "auto")
```

Note that 4-bit model serialization is currently not supported.

### Offload modules to cpu and disk 

You can offload some modules to cpu/disk if you don't have enough space on the GPU to store the entire model on your GPUs.
This uses big model inference under the hood. Check this [documentation](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) for more details. 

For 8-bit quantization, the selected modules will be converted to 8-bit precision. 

For 4-bit quantization, the selected modules will be kept in `torch_dtype` that the user passed in `BnbQuantizationConfig`.  We will add support to convert these offloaded modules in 4-bit when 4-bit serialization will be possible. 

 You just need to pass a custom `device_map` in order to offload modules on cpu/disk. The offload modules will be dispatched on the GPU when needed. Here's an example :

```py
device_map = {
    "transformer.wte": 0,
    "transformer.wpe": 0,
    "transformer.drop": 0,
    "transformer.h": "cpu",
    "transformer.ln_f": "disk",
    "lm_head": "disk",
}
```
### Fine-tune a quantized model

It is not possible to perform pure 8bit or 4bit training on these models. However, you can train these models by leveraging parameter efficient fine tuning methods (PEFT) and train for example adapters on top of them. Please have a look at [peft](https://github.com/huggingface/peft) library for more details.

Currently, you can't add adapters on top of any quantized model. However, with the official support of adapters with 🤗 Transformers models, you can fine-tune quantized models. If you want to finetune a 🤗 Transformers model , follow this [documentation](https://huggingface.co/docs/transformers/main_classes/quantization) instead. Check out this [demo](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) on how to fine-tune a 4-bit 🤗 Transformers model. 

Note that you don’t need to pass `device_map` when loading the model for training. It will automatically load your model on your GPU. Please note that `device_map=auto` should be used for inference only.

### Example demo - running GPT2 1.5b on a Google Colab

Check out the Google Colab [demo](https://colab.research.google.com/drive/1T1pOgewAWVpR9gKpaEWw4orOrzPFb3yM?usp=sharing) for running quantized models on a GTP2 model. The GPT2-1.5B model checkpoint is in FP32 which uses 6GB of memory. After quantization, it uses 1.6GB with 8-bit modules and 1.2GB with 4-bit modules.
