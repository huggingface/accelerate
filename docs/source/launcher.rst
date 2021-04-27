.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.


Notebook Launcher
=======================================================================================================================

Launch your training function inside a notebook. Currently supports launching a training with TPUs on [Google
Colab](https://colab.research.google.com/) and [Kaggle kernels](https://www.kaggle.com/code), as well as training on
several GPUs (if the machine on which you are running your notebook has them).

An example can be found in `this notebook
<https://github.com/huggingface/notebooks/blob/master/examples/accelerate/simple_nlp_example.ipynb>`__.

.. warning::

    Your :obj:`Accelerator` object should only be defined inside the training function. This is because the
    initialization should be done inside the launcher only.

.. autofunction:: accelerate.notebook_launcher
