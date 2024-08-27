# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

import torch
import transformers
from measures_util import end_measure, log_measures, start_measure
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from accelerate.utils import compute_module_sizes


DEFAULT_MODELS = {
    "gpt-j-6b": {"is_causal": True, "model": "sgugger/sharded-gpt-j-6B", "tokenizer": "EleutherAI/gpt-j-6B"},
    "gpt-neox": {"is_causal": True, "model": "EleutherAI/gpt-neox-20b"},
    "opt": {"is_causal": True, "model": "facebook/opt-30b"},
    "T0pp": {"is_causal": False, "model": "bigscience/T0pp", "model_revision": "sharded"},
}

PROMPTS = [
    "Hello, my name is",
    "Are unicorns real? Unicorns are",
    "For the first time in several years,",
    "My name is Julien and I am",
    "The goal of life is",
    "Whenever I'm sad, I like to",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run and time generations on a big model using Accelerate.")
    parser.add_argument("model_name", type=str, default=None, help="The name of the model to try.")
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="The name of the tokenizer (if different from the model."
    )
    parser.add_argument("--is_causal", type=bool, default=None, help="Whether or not the model is causal.")
    parser.add_argument(
        "--model_revision", type=str, default=None, help="The revision to use for the model checkpoint."
    )
    parser.add_argument("--torch_dtype", type=str, default=None, help="The dtype for the model.")
    parser.add_argument("--disk_offload", action="store_true")

    args = parser.parse_args()

    # Sanitize args
    if args.model_name in DEFAULT_MODELS:
        defaults = DEFAULT_MODELS[args.model_name]
        args.model_name = defaults["model"]
        if args.tokenizer_name is None:
            args.tokenizer_name = defaults.get("tokenizer", args.model_name)
        if args.is_causal is None:
            args.is_causal = defaults["is_causal"]
        if args.model_revision is None:
            args.model_revision = defaults.get("model_revision", "main")

    if args.is_causal is None:
        raise ValueError("Could not infer the default for `--is_causal`, pass either True or False for it.")
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name
    if args.model_revision is None:
        args.model_revision = "main"

    return args


def main():
    transformers.utils.logging.set_verbosity_error()
    args = parse_args()

    if args.torch_dtype is None:
        config = AutoConfig.from_pretrained(args.model_name)
        torch_dtype = getattr(config, "torch_dtype", torch.float32)
    else:
        torch_dtype = getattr(torch, args.torch_dtype)
    model_cls = AutoModelForCausalLM if args.is_causal else AutoModelForSeq2SeqLM
    kwargs = {
        "torch_dtype": torch_dtype,
        "revision": args.model_revision,
    }
    if args.disk_offload:
        kwargs["offload_folder"] = "tmp_offload"
        kwargs["offload_state_dict"] = True

    start_measures = start_measure()
    model = model_cls.from_pretrained(args.model_name, device_map="auto", **kwargs)
    end_measures = end_measure(start_measures)
    log_measures(end_measures, "Model loading")

    module_sizes = compute_module_sizes(model)
    device_size = {v: 0 for v in model.hf_device_map.values()}
    for module, device in model.hf_device_map.items():
        device_size[device] += module_sizes[module]
    message = "\n".join([f"- {device}: {size // 2**20}MiB" for device, size in device_size.items()])
    print(f"\nTheoretical use:\n{message}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    start_measures = start_measure()
    generation_times = []
    gen_tokens = []
    texts_outs = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(0)
        tokens = inputs["input_ids"][0].tolist()
        before_generate = time.time()
        outputs = model.generate(inputs["input_ids"])
        after_generate = time.time()
        outputs = outputs[0].tolist()
        num_gen_tokens = len(outputs) if outputs[: len(tokens)] != tokens else len(outputs) - len(tokens)
        generation_time = after_generate - before_generate

        text_out = tokenizer.decode(outputs, skip_special_tokens=True)
        texts_outs.append(text_out)
        generation_times.append(generation_time)
        gen_tokens.append(num_gen_tokens)
        print(f"Prompt: {prompt}\nGeneration {text_out}\nIn {generation_time:.2f}s for {num_gen_tokens} tokens\n")

    end_measures = end_measure(start_measures)
    log_measures(end_measures, "Model generation")

    generation_times_per_token = [gen / tok for gen, tok in zip(generation_times, gen_tokens)]
    avg_gen = sum(generation_times_per_token) / len(generation_times)
    print(f"Average time of generation per token: {avg_gen:.2f}s")
    print(f"First generation (avg time per token): {generation_times_per_token[0]:.2f}s")
    avg_gen = sum(generation_times_per_token[1:]) / (len(generation_times_per_token) - 1)
    print(f"Average time of generation per token (excluding the first): {avg_gen:.2f}s")


if __name__ == "__main__":
    main()
