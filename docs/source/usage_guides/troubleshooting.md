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

# Troubleshooting guide

This guide aims to provide you the tools and knowledge required to navigate some common issues. However, 
as 🤗 Accelerate continuously evolves and the use cases and setups are diverse, you might encounter an issue not covered in this 
guide. If the suggestions listed in this guide do not cover your such situation, please refer to the final section of 
the guide, [Asking for Help](#ask-for-help), to learn where to find help with your specific issue.

## Logging

When facing an error, logging can help narrows down where it is coming from. In a distributed setup with multiple processes, 
logging can be a challenge, but 🤗 Accelerate provides a utility that streamlines the logging process and ensures that 
logs are synchronized and managed effectively across the distributed setup. 

To troubleshoot an issue, use `accelerate.logging` instead of the standard Python `logging` module:

```diff
- import logging
+ from accelerate.logging import get_logger
- logger = logging.getLogger(__name__)
+ logger = get_logger(__name__)
```

To set the log level (`INFO`, `DEBUG`, `WARNING`, `ERROR`, `CRITICAL`), export it as the `ACCELERATE_LOG_LEVEL` environment,
or pass as `log_level` to `get_logger`:

```python
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")
```

By default, the log is called on main processes only. To call it on all processes, pass `main_process_only=False`.
If a log should be called on all processes and in order, also pass `in_order=True`.

## Hanging code and timeout errors

If your code seems to be hanging for a significant amount time, a common cause is mismatched shapes of tensors on different 
devices. 

When running scripts in a distributed fashion, functions such as [`Accelerator.gather`] and [`Accelerator.reduce`] are 
necessary to grab tensors across devices to perform operations on them. These (and other) functions rely on 
`torch.distributed`, which requires that tensors have the **exact same shape** across all processes for it to work.
When the tensor shapes don't match, you will experience handing code, and eventually hit a timeout exception. 

If you suspect this to be the case, use Accelerate's operational debug mode to immediately catch the issue. 

### Accelerate's operational debug mode

The recommended way to enable Accelerate's operational debug mode is during `accelerate config` setup. 
Alternative ways to enable debug mode are: 

* From the CLI: 

```
accelerate launch --debug {my_script.py} --arg1 --arg2
```

* As an environmental variable (which avoids the need for `accelerate launch`):

```
ACCELERATE_DEBUG_MODE="1" accelerate launch {my_script.py} --arg1 --arg2
```

* Manually changing the `config.yaml` file:

```diff
 compute_environment: LOCAL_MACHINE
+debug: true
```

Once you enable the debug mode, you should get a similar traceback that points to the tensor shape mismatch issue:

```
Traceback (most recent call last):
  File "/home/zach_mueller_huggingface_co/test.py", line 18, in <module>
    main()
  File "/home/zach_mueller_huggingface_co/test.py", line 15, in main
        main()broadcast_tensor = broadcast(tensor)
  File "/home/zach_mueller_huggingface_co/accelerate/src/accelerate/utils/operations.py", line 303, in wrapper
    broadcast_tensor = broadcast(tensor)
accelerate.utils.operations.DistributedOperationException: Cannot apply desired operation due to shape mismatches. All shapes across devices must be valid.

Operation: `accelerate.utils.operations.broadcast`
Input shapes:
  - Process 0: [1, 5]
  - Process 1: [1, 2, 5]
```

## Non-reproducible results between device setups

https://huggingface.co/docs/accelerate/concept_guides/performance

## Performance issues on different GPUs

If your milt-GPU setup consists of different GPUs, you may hit some limitations:

- There may be imbalance in GPU memory between the GPUs. In this case, the GPU with smaller memory will limit the batch size or the size of the model that can be loaded onto the GPUs.
- If you are using GPUs with different performance profiles, the performance will be driven by the slowest GPU that you are using as the other GPUs will have to wait for it to complete its workload.

Vastly different GPUs within the same setup can lead to performance bottlenecks. 

## Ask for help

If the above troubleshooting tools and advice did not help you resolve your issue, reach out for help to the community 
and the team.

### Forums 

Ask for help on the Hugging Face forums - post your question in the [🤗Accelerate category](https://discuss.huggingface.co/c/accelerate/18) 
Make sure to write a descriptive post with relevant context about your setup and reproducible code to maximize the likelihood that your problem is solved!

### Discord

Post a question on [Discord](http://hf.co/join/discord), and let the team and the community help you.

### GitHub Issues

Create an Issue on the 🤗 Accelerate repository if you suspect to have found a bug related to the library. Try to include 
as much context regarding the bug and your distributed setup as possible to help us better figure out what's wrong and how we can fix it.
