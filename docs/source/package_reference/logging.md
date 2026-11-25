<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Logging

Refer to the [Troubleshooting guide](../basic_tutorials/troubleshooting#logging) or to the example below to learn 
how to use Accelerate's logger. 

[[autodoc]] logging.get_logger

## Library verbosity

Use [`logging.set_verbosity`] to set the default level for Accelerate's internal loggers:

```python
from accelerate.logging import set_verbosity

set_verbosity("warning")
```

Per-logger levels set through `log_level` or `ACCELERATE_LOG_LEVEL` take precedence.

[[autodoc]] logging.get_verbosity

[[autodoc]] logging.set_verbosity

[[autodoc]] logging.set_verbosity_debug

[[autodoc]] logging.set_verbosity_info

[[autodoc]] logging.set_verbosity_warning

[[autodoc]] logging.set_verbosity_error
