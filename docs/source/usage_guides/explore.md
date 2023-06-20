<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Learning how to incorporate 🤗 Accelerate features quickly!

Please use the interactive tool below to help you get started with learning about a particular 
feature of 🤗 Accelerate and how to utilize it! It will provide you with a code diff, an explaination
towards what is going on, as well as provide you with some useful links to explore more within
the documentation!

Most code examples start from the following python code before integrating 🤗 Accelerate in some way:

```python
for batch in dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

<div class="block dark:hidden">
	<iframe 
        src="https://muellerzr-accelerate-examples.hf.space?__theme=light"
        width="850"
        height="1600"
    ></iframe>
</div>
<div class="hidden dark:block">
    <iframe 
        src="https://muellerzr-accelerate-examples.hf.space?__theme=dark"
        width="850"
        height="1600"
    ></iframe>
</div>
