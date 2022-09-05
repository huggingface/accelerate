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
from abc import ABC
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available
from .operations import recursively_apply


if is_megatron_lm_available():
    from megatron import get_args, mpu
    from megatron.arguments import _add_data_args
    from megatron.data.data_samplers import build_pretraining_data_loader
    from megatron.data.dataset_utils import build_train_valid_test_datasets
    from megatron.initialize import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_src_rank,
        initialize_megatron,
        set_jit_fusion_options,
    )
    from megatron.model import BertModel, GPTModel, ModelType, T5Model
    from megatron.model.classification import Classification
    from megatron.model.module import MegatronModule
    from megatron.optimizer import get_megatron_optimizer
    from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    from megatron.training import build_train_valid_test_data_iterators, get_model, get_optimizer_param_scheduler
    from megatron.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids


def model_provider_func(accelerator, pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the model."""
    args = get_args()
    mode = "pre-training" if args.pretraining_flag else "fine-tuning"
    accelerator.print(f"Building {args.model_type_name} model in the {mode} mode.")
    if args.model_type_name == "bert":
        if args.pretraining_flag:
            num_tokentypes = 2 if args.bert_binary_head else 0
            model = BertModel(
                num_tokentypes=num_tokentypes,
                add_binary_head=args.bert_binary_head,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
            )
        else:

            model = Classification(
                num_classes=args.num_labels, num_tokentypes=2, pre_process=pre_process, post_process=post_process
            )
    elif args.model_type_name == "gpt2":
        model = GPTModel(num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process)
    elif args.model_type_name == "t5":
        model = T5Model(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type_name}")
    return model


def prepare_data_loader(accelerator, dataloader, consumed_samples_index=-1, consumed_samples=0):
    accelerator.print("Preparing dataloader")
    args = get_args()
    if not args.megatron_dataset_flag:
        if args.consumed_samples is not None:
            consumed_samples = args.consumed_samples[consumed_samples_index]
        return build_pretraining_data_loader(dataloader.dataset, consumed_samples)
    else:
        if args.consumed_samples is not None:
            (
                args.consumed_train_samples,
                args.consumed_valid_samples,
                args.consumed_test_samples,
            ) = args.consumed_samples
        else:
            args.consumed_train_samples, args.consumed_valid_samples, args.consumed_test_samples = 0, 0, 0
        train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
            MegatronLMDummyDataLoader.get_train_valid_test_datasets_provider
        )
        return train_data_iterator, valid_data_iterator, test_data_iterator


def prepare_model(accelerator):
    accelerator.print("Preparing model")
    args = get_args()
    if args.model_type_name == "bert" or args.model_type_name == "gpt":
        model_type = ModelType.encoder_or_decoder
    elif args.model_type_name == "t5":
        model_type = ModelType.encoder_and_decoder
    model = get_model(model_provider_func, model_type)
    return model


def prepare_optimizer(accelerator, model):
    accelerator.print("Preparing optimizer")
    args = get_args()
    optimizer = get_megatron_optimizer(model, args.no_wd_decay_cond, args.scale_lr_cond, args.lr_mult)
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
    args = get_args()
    args.padded_vocab_size = _vocab_size_with_padding(args.orig_vocab_size, args)
    if args.model_type_name == "bert" and args.pretraining_flag and args.num_labels == 2:
        args.bert_binary_head = True
    # if args.virtual_pipeline_model_parallel_size is not None:
    #     raise Exception("Virtual pipeline model parallelism is not supported in Accelerate")


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


class MegatronLMDummyDataLoader:
    """
    Dummy dataloader presents model parameters or param groups, this is primarily used to follow conventional training

    Args:
        **dataset_kwargs: Megatron data arguments.
    """

    def __init__(self, **dataset_kwargs):
        parser = argparse.ArgumentParser()
        parser = _add_data_args(parser)
        data_args = parser.parse_known_args()
        self.dataset_args = vars(data_args[0])
        self.dataset_args.update(dataset_kwargs)
        self.dataset_args["megatron_dataset_flag"] = True

    def set_megatron_data_args(self):
        args = get_args()
        for key, value in self.dataset_args.items():
            setattr(args, key, value)

    @staticmethod
    def get_train_valid_test_datasets_provider():
        def train_valid_test_datasets_provider(train_val_test_num_samples):
            """Build train, valid, and test datasets."""
            args = get_args()
            if args.model_type_name == "bert":
                dataset_args = {
                    "data_path": args.data_path,
                    "data_impl": args.data_impl,
                    "splits_string": args.split,
                    "train_valid_test_num_samples": train_val_test_num_samples,
                    "max_seq_length": args.seq_length,
                    "masked_lm_prob": args.mask_prob,
                    "short_seq_prob": args.short_seq_prob,
                    "skip_warmup": (not args.mmap_warmup),
                    "binary_head": args.bert_binary_head,
                    "seed": args.seed,
                }
            elif args.model_type_name == "gpt":
                dataset_args = {
                    "data_path": args.data_path,
                    "data_impl": args.data_impl,
                    "splits_string": args.split,
                    "train_valid_test_num_samples": train_val_test_num_samples,
                    "seq_length": args.seq_length,
                    "skip_warmup": (not args.mmap_warmup),
                    "seed": args.seed,
                }
            elif args.model_type_name == "t5":
                dataset_args = {
                    "data_path": args.data_path,
                    "data_impl": args.data_impl,
                    "splits_string": args.split,
                    "train_valid_test_num_samples": train_val_test_num_samples,
                    "max_seq_length": args.encoder_seq_length,
                    "max_seq_length_dec": args.decoder_seq_length,
                    "masked_lm_prob": args.mask_prob,
                    "short_seq_prob": args.short_seq_prob,
                    "skip_warmup": (not args.mmap_warmup),
                    "dataset_type": "t5",
                    "seed": args.seed,
                }
            else:
                raise ValueError(f"Unknown model type: {args.model_type_name}")
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(**dataset_args)
            return train_ds, valid_ds, test_ds

        return train_valid_test_datasets_provider


class MegatronLMModelWrapper(MegatronModule):
    """
    Megatron-LM model wrapper
    """

    def __init__(self, model, optimizer, scheduler):
        self.module = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    # args = get_args()

    def train_step():
        pass

    def train(self):
        for model_module in self.module:
            model_module.train()

    def eval(self):
        for model_module in self.module:
            model_module.eval()

    def forward(self, *inputs, **kwargs):
        pass


class MegatronLMOptimizerWrapper(AcceleratedOptimizer):
    def __init__(self, optimizer):
        super().__init__(optimizer, device_placement=False, scaler=None)

    def zero_grad(self, set_to_none=None):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed

    def step(self):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        return self.optimizer.overflow


class MegatronLMSchedulerWrapper(AcceleratedScheduler):
    def __init__(self, scheduler, optimizers):
        super().__init__(scheduler, optimizers)

    def step(self):
        pass  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed


class AbstractTrainStep(ABC):
    """Abstract class for batching, forwardPass and loss handler."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_batch_func(self):
        pass

    def get_forward_step_func(self):
        pass

    def get_loss_func(self):
        pass


class BertTrainStep(AbstractTrainStep):
    """Bert train step class."""

    def __init__(self):
        super().__init__("BertTrainStep")
        args = get_args()
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func(args.pretraining_flag, args.num_labels)
        self.forward_step = self.get_forward_step_func(args.pretraining_flag, args.bert_binary_head)

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch_megatron(data_iterator):
            """Build the batch."""

            # Items and their type.
            keys = ["text", "types", "labels", "is_random", "loss_mask", "padding_mask"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens = data_b["text"].long()
            types = data_b["types"].long()
            sentence_order = data_b["is_random"].long()
            loss_mask = data_b["loss_mask"].float()
            lm_labels = data_b["labels"].long()
            padding_mask = data_b["padding_mask"].long()

            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
                data["loss_mask"] = (data["labels"] != -100).to(torch.float)
            else:
                data = None
            data_b = broadcast_data(data)

            # Unpack.
            tokens = data_b["input_ids"].long()
            padding_mask = data_b["attention_mask"].long()
            if "token_type_ids" in data_b:
                types = data_b["token_type_ids"].long()
            else:
                types = None
            if "labels" in data_b:
                lm_labels = data_b["labels"].long()
            else:
                lm_labels = None
            if "next_sentence_label" in data_b:
                sentence_order = data_b["next_sentence_label"].long()
            else:
                sentence_order = None
            loss_mask = data_b["loss_mask"].float()

            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self, pretraining_flag, num_labels):
        def loss_func_pretrain(loss_mask, sentence_order, output_tensor):
            lm_loss_, sop_logits = output_tensor

            lm_loss_ = lm_loss_.float()
            loss_mask = loss_mask.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

            if sop_logits is not None:
                sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
                sop_loss = sop_loss.float()
                loss = lm_loss + sop_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss, sop_loss])
                return loss, {"lm loss": averaged_losses[0], "sop loss": averaged_losses[1]}

            else:
                loss = lm_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss])
                return loss, {"lm loss": averaged_losses[0]}

        def loss_func_finetune(labels, logits):
            if num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            averaged_losses = average_losses_across_data_parallel_group([loss])
            return loss, {"loss": averaged_losses[0]}

        if pretraining_flag:
            return loss_func_pretrain
        else:
            return loss_func_finetune

    def get_forward_step_func(self, pretraining_flag, bert_binary_head):
        def forward_step(data_iterator, model):
            """Forward step."""
            tokens, types, sentence_order, loss_mask, labels, padding_mask = self.get_batch(data_iterator)
            if not bert_binary_head:
                types = None
            # Forward pass through the model.
            if pretraining_flag:
                output_tensor = model(tokens, padding_mask, tokentype_ids=types, lm_labels=labels)
                return output_tensor, partial(self.loss_func, loss_mask, sentence_order)
            else:
                logits = model(tokens, padding_mask, tokentype_ids=types)
                return logits, partial(self.loss_func, labels)

        return forward_step


class GPTTrainStep(AbstractTrainStep):
    def __init__(self):
        super().__init__("GPTTrainStep")
        args = get_args()
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch_megatron(data_iterator):
            from megatron import get_tokenizer

            """Generate a batch"""
            args = get_args()
            tokenizer = get_tokenizer()

            # Items and their type.
            keys = ["text"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens_ = data_b["text"].long()
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()

            # Get the masks and postition ids.
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens, tokenizer.eod, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss
            )

            return tokens, labels, loss_mask, attention_mask, position_ids

        def get_batch_transformer(data_iterator):
            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
                tokens_ = data["input_ids"].long()
                labels = tokens_[:, 1:].contiguous()
                tokens = tokens_[:, :-1].contiguous()
                # Get the masks and postition ids.
                attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                    tokens, 0, False, False, False
                )
                data["input_ids"] = tokens
                data["labels"] = labels
                data["loss_mask"] = loss_mask
                data["attention_mask"] = attention_mask
                data["position_ids"] = position_ids
            else:
                data = None
            data_b = broadcast_data(data)
            tokens = data_b["input_ids"]
            labels = data_b["labels"]
            loss_mask = data_b["loss_mask"]
            attention_mask = data_b["attention_mask"]
            position_ids = data_b["position_ids"]
            return tokens, labels, loss_mask, attention_mask, position_ids

        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):
        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {"lm loss": averaged_loss[0]}

        return loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.loss_func, loss_mask)

        return forward_step


class T5TrainStep(AbstractTrainStep):
    pass


def broadcast_data(data):
    def _gpu_broadcast_one(tensor, src=0, group=None):
        torch.distributed.broadcast(tensor, src=src, group=group)
        return tensor

    return recursively_apply(
        _gpu_broadcast_one,
        data,
        error_on_other_type=True,
        src=get_tensor_model_parallel_src_rank(),
        group=get_tensor_model_parallel_group(),
    )
