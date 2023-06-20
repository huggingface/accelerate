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

# Logging with Accelerate

Accelerate has its own logging utility to handle logging while in a distributed system.
To utilize this replace cases of `logging` with `accelerate.logging`:
```diff
- import logging
+ from accelerate.logging import get_logger
- logger = logging.getLogger(__name__)
+ logger = get_logger(__name__)
```

## Setting the log level

The log level can be set with the `ACCELERATE_LOG_LEVEL` environment variable or by passing 
`log_level` to `get_logger`:
```python
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")
```

[[autodoc]] logging.get_logger