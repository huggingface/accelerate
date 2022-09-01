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

import torch

from .imports import is_megatron_lm_available


if is_megatron_lm_available():
    from megatron import get_args
    from megatron.data.data_samplers import build_pretraining_data_loader
    from megatron.initialize import initialize_megatron, set_jit_fusion_options
    from megatron.model import BertModel, ModelType
    from megatron.model.classification import Classification
    from megatron.optimizer import get_megatron_optimizer
    from megatron.training import get_model, get_optimizer_param_scheduler


def model_provider_func(accelerator, pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the model."""
    args = get_args()
    mode = "pre-training" if args.pretraining_flag else "fine-tuning"
    accelerator.print(f"Building {args.model_type_name} model in the {mode} mode.")
    if args.model_type_name == "bert":
        if args.pretraining_flag:

            bert_binary_head = args.num_labels == 2
            num_tokentypes = 2 if bert_binary_head else 0
            model = BertModel(
                num_tokentypes=num_tokentypes,
                add_binary_head=bert_binary_head,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
            )
        else:

            model = Classification(
                num_classes=args.num_labels, num_tokentypes=2, pre_process=pre_process, post_process=post_process
            )
    elif args.model_type_name == "gpt2":
        from megatron.model import GPTModel

        model = GPTModel(num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process)
    elif args.model_type_name == "t5":
        from megatron.model import T5Model

        model = T5Model(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
        )
    return model


def prepare_data_loader(accelerator, dataset, consumed_samples=0):
    accelerator.print("Preparing dataloader")
    return build_pretraining_data_loader(dataset, consumed_samples)


def prepare_model(accelerator):
    accelerator.print("Preparing model")
    args = get_args()
    if args.model_type_name == "bert" or args.model_type_name == "gpt":
        model_type = ModelType.encoder_or_decoder
    elif args.model_type_name == "t5":
        model_type = ModelType.encoder_and_decoder
    model = get_model(model_provider_func, model_type)
    return model


def prepare_optimizer(accelerator, model, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):

    accelerator.print("Preparing optimizer")
    optimizer = get_megatron_optimizer(model, no_wd_decay_cond, scale_lr_cond, lr_mult)
    return optimizer


def prepare_scheduler(accelerator, optimizer, scheduler, is_dummy_scheduler):
    accelerator.print("Preparing scheduler")
    if is_dummy_scheduler:
        scheduler = get_optimizer_param_scheduler(optimizer)
    else:
        scheduler.optimizer = optimizer
        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler = scheduler.__class__(optimizer, scheduler.lr_lambdas[0])
    return scheduler


def initialize(accelerator, extra_args_provider=None, args_defaults={}):
    accelerator.print("Initializing Megatron-LM")
    # Initalize and get arguments
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=args_defaults, ignore_unknown_args=True)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()


class MegatronLMDummyScheduler:
    """
    Dummy scheduler presents model parameters or param groups, this is primarily used to follow conventional training
    loop when scheduler config is specified in the deepspeed config file.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        total_num_steps (int):
            Total number of steps.
        warmup_num_steps (int):
            Number of steps for warmup.
        **kwargs:
            Other arguments.
    """

    def __init__(self, optimizer, total_num_steps=None, warmup_num_steps=0, **kwargs):
        self.optimizer = optimizer
        self.total_num_steps = total_num_steps
        self.warmup_num_steps = warmup_num_steps
        self.kwargs = kwargs


class MegatronLMModelWrapper:
    pass


class MegatronLMOptimizerWrapper:
    pass


class MegatronLMSchedulerWrapper:
    pass
