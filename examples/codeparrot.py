# Training script based on TRL, but using just accelerate
# original script: https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
import time

model_name = "bigscience/bloomz-560m"
dataset_name = "timdettmers/openassistant-guanaco"
dataset_text_field = "text"
learning_rate = 1.41e-5
batch_size = 8
max_seq_length = 256
gradient_accumulation_steps = 2
peft_lora_r = 64
peft_lora_alpha = 16
num_training_steps=200


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

from torch import nn
def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output

children = nested_children(model)

from transformer_engine.pytorch.module.layernorm import LayerNorm as te_LayerNorm
import transformer_engine.pytorch as te

def replace_layers(model, keys, replace_func):
    current_module = model

    for key in keys[:-1]:
        if isinstance(key, int):
            current_module = current_module[key]
        elif isinstance(current_module, nn.Module) and key in current_module._modules:
            current_module = current_module._modules[key]
        elif hasattr(current_module, key):
            current_module = getattr(current_module, key)
        else:
            raise KeyError(f"Key '{key}' not found in the model.")

    last_key = keys[-1]
    if isinstance(last_key, int):
        current_module[last_key] = replace_func(current_module[last_key])
    elif isinstance(current_module, nn.Module) and last_key in current_module._modules:
        current_module._modules[last_key] = replace_func(current_module._modules[last_key])
    elif hasattr(current_module, last_key):
        setattr(current_module, last_key, replace_func(getattr(current_module, last_key)))
    else:
        raise KeyError(f"Key '{last_key}' not found in the model.")

    return model

def check_for_layer(dictionary, layer, current_key=None):
    if current_key is None:
        current_key = []
    matching_keys = []

    for k, v in dictionary.items():
        if isinstance(v, layer):
            matching_keys.append(".".join(current_key + [k]))
        elif isinstance(v, dict):
            matching_keys.extend(check_for_layer(v, layer, current_key + [k]))

    return matching_keys

# First LayerNorm
def replace_layernorm(module):
    te_module = te_LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
    module.weight.copy_(te_module.weight)
    module.bias.copy_(te_module.bias)
    return te_module

layernorm_locations = check_for_layer(children, nn.LayerNorm)
print(f'Num layernorm: {len(layernorm_locations)}')
# then nn.Linear
def replace_linear(module):
    # Return early if the linear layer weights are not multiples of 16
    if any(p % 16 != 0 for p in module.weight.shape):
        return module
    has_bias = module.bias is not None
    te_module = te.Linear(
        module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
    )
    module.weight.copy_(te_module.weight)
    if has_bias:
        module.bias.copy_(te_module.bias)
    return te_module
linear_locations = check_for_layer(children, nn.Linear)
print(f'Num linear: {len(linear_locations)}')

with torch.no_grad():
    for location in layernorm_locations:
        model = replace_layers(model, location.split('.'), replace_func=replace_layernorm)

    for location in linear_locations:
        model = replace_layers(model, location.split('.'), replace_func=replace_linear)
    


def get_dataloaders(batch_size:int = 8):
    dataset = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
    
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )

    pad_to_multiple_of = None
    pad_to_multiple_of = 16


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader_params = {
        "batch_size": batch_size, 
        "collate_fn": data_collator,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset, **dataloader_params)
    return train_dataloader


train_dataloader = get_dataloaders(batch_size)

optimizer = AdamW(params = model.parameters(), lr=learning_rate)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

accelerator = Accelerator(mixed_precision="fp8")

model, optimizer = accelerator.prepare(model, optimizer)

model.train()
completed_steps = 0
total_loss = 0
start_time = time.time()
# 100 just to get the full time in
batch_times = []
model = model.to("cuda")
for _ in range(100):
    if completed_steps >= num_training_steps:
        break   
    for step, batch in enumerate(train_dataloader):
        batch = batch.to("cuda")
        outputs = model(**batch)
        loss = outputs.loss
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
        
        end_time = time.time()
        total_time = end_time - start_time 
        batch_times.append(total_time)
        start_time = end_time

        if completed_steps >= num_training_steps:
            break
    
print(f"Average time per batch: {sum(batch_times) / len(batch_times)}")

from accelerate.utils import convert_bytes
print(f'Maximum memory allocated: {convert_bytes(torch.cuda.max_memory_allocated())}')

