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
`pip install bitsandbytes==0.39.0`
- Install latest `accelerate` from source
`pip install git+https://github.com/huggingface/accelerate.git`

### How it works

First, we need to initialize our model. To save memory, we can initialize an empty model using the context manager [init_empty_weights]

```py
from accelerate import init_empty_weights

with init_empty_weights():
    empty_model = ModelClass(...)
```

Then, we need to get the path to the weights of your model. The path can be the state_dict file (e.g. "pytorch_model.bin") or a folder containing the sharded checkpoints. 

```py
weights_location = ...
```

Finally, you need to set your quantization configuration with ['~utils.BnbQuantizationConfig'].

Here's an example for 8-bit quantization:
```py
from accelerate.utils import BnbQuantizationConfig
quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
```

Here's an example for 4-bit quantization:
```py
from accelerate.utils import BnbQuantizationConfig
quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
```

To quantize your empty model with the selected configuration, you need to use [`~utils.load_and_quantize`]. 

```py
from accelerate.utils import load_and_quantize
quantized_model = load_and_quantize(empty_model, weight_location=weight_location, quantization_config=quantization_config, device_map = "auto")
```

### Saving and loading 8-bit model

You can save your 8-bit model with accelerate using [`~Accelerator.save_model`]. 

```py
from accelerate import Accelerator
accelerate = Accelerator()
new_weight_location = "path/to/save_directory"
accelerate.save_model(quantized_model, new_weight_location)

quantized_model_from_saved = load_and_quantize(empty_model, weight_location=new_weight_location, quantization_config=quantization_config, device_map = "auto")
```

Note that 4-bit model serialization is currently not supported.

### Fine-tune a quantized model

With the official support of adapters in the Hugging Face ecosystem, you can fine-tune quantized models. Please have a look at [peft](https://github.com/huggingface/peft) library for more details.

Note that you don’t need to pass `device_map` when loading the model for training. It will automatically load your model on your GPU. Please note that `device_map=auto` should be used for inference only.

### Example demo - running GPT2 1.5b on a Google Colab

Check out the Google Colab [demo](https://colab.research.google.com/drive/1T1pOgewAWVpR9gKpaEWw4orOrzPFb3yM?usp=sharing) for running quantized models on a GTP2 model. The GPT2-1.5B model checkpoint is in FP32 which uses 6GB of memory. After quantization, it uses 1.5GB with 8-bit modules and 0.9GB with 4-bit modules.