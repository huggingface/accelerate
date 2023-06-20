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

# Launching your 🤗 Accelerate scripts

In the previous tutorial, you were introduced to how to modify your current training script to use 🤗 Accelerate.
The final version of that code is shown below:

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

But how do you run this code and have it utilize the special hardware available to it?

First, you should rewrite the above code into a function, and make it callable as a script. For example:

```diff
  from accelerate import Accelerator
  
+ def main():
      accelerator = Accelerator()

      model, optimizer, training_dataloader, scheduler = accelerator.prepare(
          model, optimizer, training_dataloader, scheduler
      )

      for batch in training_dataloader:
          optimizer.zero_grad()
          inputs, targets = batch
          outputs = model(inputs)
          loss = loss_function(outputs, targets)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()

+ if __name__ == "__main__":
+     main()
```

Next, you need to launch it with `accelerate launch`. 

<Tip warning={true}>

  It's recommended you run `accelerate config` before using `accelerate launch` to configure your environment to your liking. 
  Otherwise 🤗 Accelerate will use very basic defaults depending on your system setup.

</Tip>


## Using accelerate launch

🤗 Accelerate has a special CLI command to help you launch your code in your system through `accelerate launch`.
This command wraps around all of the different commands needed to launch your script on various platforms, without you having to remember what each of them is.

<Tip>

  If you are familiar with launching scripts in PyTorch yourself such as with `torchrun`, you can still do this. It is not required to use `accelerate launch`.

</Tip>

You can launch your script quickly by using:

```bash
accelerate launch {script_name.py} --arg1 --arg2 ...
```

Just put `accelerate launch` at the start of your command, and pass in additional arguments and parameters to your script afterward like normal!

Since this runs the various torch spawn methods, all of the expected environment variables can be modified here as well.
For example, here is how to use `accelerate launch` with a single GPU:

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...
```

You can also use `accelerate launch` without performing `accelerate config` first, but you may need to manually pass in the right configuration parameters.
In this case, 🤗 Accelerate will make some hyperparameter decisions for you, e.g., if GPUs are available, it will use all of them by default without the mixed precision.
Here is how you would use all GPUs and train with mixed precision disabled:

```bash
accelerate launch --multi_gpu {script_name.py} {--arg1} {--arg2} ...
```

Or by specifying a number of GPUs to use:

```bash
accelerate launch --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

To get more specific you should pass in the needed parameters yourself. For instance, here is how you 
would also launch that same script on two GPUs using mixed precision while avoiding all of the warnings: 

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

For a complete list of parameters you can pass in, run:

```bash
accelerate launch -h
```

<Tip>

  Even if you are not using 🤗 Accelerate in your code, you can still use the launcher for starting your scripts!

</Tip>

For a visualization of this difference, that earlier `accelerate launch` on multi-gpu would look something like so with `torchrun`:

```bash
MIXED_PRECISION="fp16" torchrun --nproc_per_node=2 --num_machines=1 {script_name.py} {--arg1} {--arg2} ...
```

You can also launch your script utilizing the launch CLI as a python module itself, enabling the ability to pass in other python-specific
launching behaviors. To do so, use `accelerate.commands.launch` instead of `accelerate launch`:

```bash
python -m accelerate.commands.launch --num_processes=2 {script_name.py} {--arg1} {--arg2}
```

If you want to execute the script with any other python flags, you can pass them in as well similar to `-m`, such as 
the below example enabling unbuffered stdout and stderr:

```bash
python -u -m accelerate.commands.launch --num_processes=2 {script_name.py} {--arg1} {--arg2}
```


## Why you should always use `accelerate config`

Why is it useful to the point you should **always** run `accelerate config`? 

Remember that earlier call to `accelerate launch` as well as `torchrun`?
Post configuration, to run that script with the needed parts you just need to use `accelerate launch` outright, without passing anything else in:

```bash
accelerate launch {script_name.py} {--arg1} {--arg2} ...
```


## Custom Configurations

As briefly mentioned earlier, `accelerate launch` should be mostly used through combining set configurations 
made with the `accelerate config` command. These configs are saved to a `default_config.yaml` file in your cache folder for 🤗 Accelerate. 
This cache folder is located at (with decreasing order of priority):

- The content of your environment variable `HF_HOME` suffixed with `accelerate`.
- If it does not exist, the content of your environment variable `XDG_CACHE_HOME` suffixed with
  `huggingface/accelerate`.
- If this does not exist either, the folder `~/.cache/huggingface/accelerate`.

To have multiple configurations, the flag `--config_file` can be passed to the `accelerate launch` command paired 
with the location of the custom yaml. 

An example yaml may look something like the following for two GPUs on a single machine using `fp16` for mixed precision:
```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

Launching a script from the location of that custom yaml file looks like the following:
```bash
accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...
```
