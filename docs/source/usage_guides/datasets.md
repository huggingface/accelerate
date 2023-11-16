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

# Distributed datasets mapping with 🤗 Accelerate

Distributed mapping a function which to run over a dataset is a common use case, especially with natural language processing (NLP) models and datasets.  Typically one needs to augment a dataset with extra columns coming from some expensive operation.

## The Problem

Sometimes one needs to augment a dataset with extra columns coming from some expensive operation.  For example, one might want to add a column to a dataset with logits or embeddings from a model.  In such a usecase one might want to use a distributed mapping function to speed up the process.

## The Solution

🤗 Accelerate provides a way to distribute the dataset and mapping model via the `prepare` method.  This method will wrap the dataset and model with the appropriate distributed wrappers.  

```python
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset

def map_to_dataset(batch):
    # do some expensive operation using the model
    value = model(batch["input"])
    return value

dataset = load_dataset("my example")
dataloader = DataLoader(dataset, batch_size=8)
model = MyModel()

accelerator = Accelerator()

# prepare the dataset and model
dataloader, model = accelerator.prepare(dataloader, model)

output_values = []
for batch in dataloader:
    value = map_to_dataset(batch)

    # gather the value from all processes
    value = accelerator.gather_for_metrics(value)
    
    # each processor will have a list of values
    output_values.append(value)

final_output = torch.cat(output_values).cpu().numpy()
dataset.add_column("new_column", final_output)
```
