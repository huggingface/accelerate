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
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device


if is_transformers_available():
    from transformers.modeling_outputs import (
        CausalLMOutputWithCrossAttentions,
        Seq2SeqLMOutput,
        SequenceClassifierOutput,
    )
    from transformers.models.bert.modeling_bert import BertForPreTrainingOutput


if is_megatron_lm_available():
    from megatron import get_args, get_num_microbatches, get_timers, mpu
    from megatron.arguments import _add_data_args, parse_args, validate_args
    from megatron.checkpointing import load_args_from_checkpoint
    from megatron.data.data_samplers import build_pretraining_data_loader
    from megatron.data.dataset_utils import build_train_valid_test_datasets
    from megatron.global_vars import set_global_variables
    from megatron.initialize import _compile_dependencies, _init_autoresume, _set_random_seed, set_jit_fusion_options
    from megatron.model import BertModel, GPTModel, ModelType, T5Model
    from megatron.model.classification import Classification
    from megatron.mpu.initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_src_rank
    from megatron.optimizer import get_megatron_optimizer
    from megatron.schedules import get_forward_backward_func
    from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    from megatron.training import build_train_valid_test_data_iterators, get_model, get_optimizer_param_scheduler
    from megatron.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids


def model_provider_func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the model."""
    args = get_args()
    mode = "pre-training" if args.pretraining_flag else "fine-tuning"
    if args.rank == 0:
        print(f"Building {args.model_type_name} model in the {mode} mode.")
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
    elif args.model_type_name == "gpt":
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
        collate_fn = dataloader.collate_fn
        if args.consumed_samples is not None:
            consumed_samples = args.consumed_samples[consumed_samples_index]
        dataloader = build_pretraining_data_loader(dataloader.dataset, consumed_samples)
        dataloader.collate_fn = collate_fn
        return dataloader
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


def prepare_scheduler(accelerator, optimizer, scheduler):
    accelerator.print("Preparing scheduler")
    scheduler = get_optimizer_param_scheduler(optimizer)
    return scheduler


def initialize(accelerator, extra_args_provider=None, args_defaults={}):
    accelerator.print("Initializing Megatron-LM")
    assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args=True)

    # Set defaults
    for key, value in args_defaults.items():
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print(
                    "WARNING: overriding default arguments for {key}:{v} \
                        with {key}:{v2}".format(
                        key=key, v=getattr(args, key), v2=value
                    ),
                    flush=True,
                )
        setattr(args, key, value)

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    validate_args(args)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        device_count = torch.cuda.device_count()
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device

            # Set the tensor model-parallel, pipeline model-parallel, and
            # data-parallel communicators.
            if mpu.model_parallel_is_initialized():
                print("model parallel is already initialized")
            else:
                mpu.initialize_model_parallel(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                )

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    args = get_args()

    # Megatron's MPU is the master. Complete initialization right away.
    finish_mpu_init()

    # Autoresume.
    _init_autoresume()

    # Compile dependencies.
    _compile_dependencies()

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


class MegatronEngine(torch.nn.Module):
    """
    Megatron-LM model wrapper
    """

    def __init__(self, model, optimizer, scheduler):
        super(MegatronEngine, self).__init__()
        self.module = model
        self.base_model = model[0]
        self.optimizer = optimizer
        self.scheduler = scheduler
        args = get_args()
        if args.model_type_name == "bert":
            self.train_step_handler = BertTrainStep(args)
        elif args.model_type_name == "gpt":
            self.train_step_handler = GPTTrainStep(args)
        elif args.model_type_name == "t5":
            self.train_step_handler = T5TrainStep(args)
        else:
            raise ValueError(f"Unknown model type: {args.model_type_name}")
        self.optimizer.skipped_iter = False

    def train(self):
        for model_module in self.module:
            model_module.train()

    def eval(self):
        for model_module in self.module:
            model_module.eval()

    def train_step(self, **batch_data):
        args = get_args()
        timers = get_timers()
        if len(self.module) > 1:
            batch_data_iterator = [iter([batch_data]) for _ in range(len(self.module))]
        else:
            batch_data_iterator = iter([batch_data])

        # Set grad to zero.
        if args.DDP_impl == "local" and args.use_contiguous_buffers_in_local_ddp:
            for partition in self.module:
                partition.zero_grad_buffer()
        self.optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            self.train_step_handler.forward_step,
            batch_data_iterator,
            self.module,
            self.optimizer,
            None,
            forward_only=False,
        )

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Reduce gradients.
        timers("backward-reduce-model-grads").start()
        self.optimizer.reduce_model_grads(args, timers)
        timers("backward-reduce-model-grads").stop()

        # Update parameters.
        timers("optimizer").start()
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step(args, timers)
        timers("optimizer").stop()

        # Gather params.
        if update_successful:
            timers("backward-gather-model-params").start()
            self.optimizer.gather_model_params(args, timers)
            timers("backward-gather-model-params").stop()

        # Update learning rate.
        if update_successful:
            if self.scheduler is not None:
                increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
                self.scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        self.optimizer.skipped_iter = not update_successful

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        args.consumed_train_samples += (
            mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def eval_step(self, **batch_data):
        args = get_args()
        if len(self.module) > 1:
            batch_data_iterator = [iter([batch_data]) for _ in range(len(self.module))]
        else:
            batch_data_iterator = iter([batch_data])
        forward_backward_func = get_forward_backward_func()
        loss_dicts = forward_backward_func(
            self.train_step_handler.forward_step,
            batch_data_iterator,
            self.module,
            optimizer=None,
            timers=None,
            forward_only=True,
        )
        # Empty unused memory
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        args.consumed_valid_samples += (
            mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in loss_dicts[0]:
                losses_reduced_for_key = [x[key] for x in loss_dicts]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced
        else:
            return {}

    def forward(self, **batch_data):
        # During training, we use train_step()
        # model(**batch_data) performs following operations by delegating it to `self.train_step`:
        # 1. Prepare **batch_data for Tendor, Pipeline and Model Parallelism
        # 2. Set grad to zero.
        # 3. forward pass and backward pass using Pipeline Parallelism
        # 4. Empty unused memory.
        # 5. Reduce gradients.
        # 6. Update parameters.
        # 7. Gather params when using Distributed Optimizer (Data Parallelism).
        # 8. Update learning rate if scheduler is specified.
        # 9. Empty unused memory.
        # 10. Average loss across microbatches and across DP ranks.
        #
        # During evaluation, we use eval_step()
        args = get_args()
        if self.module[0].training:
            loss_dict, _, _, _ = self.train_step(**batch_data)
        else:
            loss_dict = self.eval_step(**batch_data)
        loss = torch.tensor(0.0, device=args.local_rank)
        for key in loss_dict:
            loss += loss_dict[key]
        # loss = reduce(loss)
        if self.train_step_handler.model_output_class is not None:
            return self.train_step_handler.model_output_class(loss=loss)
        return loss


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
        return self.optimizer.skipped_iter


class MegatronLMSchedulerWrapper(AcceleratedScheduler):
    def __init__(self, scheduler, optimizers):
        super().__init__(scheduler, optimizers)

    def step(self, *args, **kwargs):
        return  # `model(**batch)` is doing that automatically. Therefore, it's implementation is not needed


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

    def __init__(self, args):
        super().__init__("BertTrainStep")
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func(args.pretraining_flag, args.num_labels)
        self.forward_step = self.get_forward_step_func(args.pretraining_flag, args.bert_binary_head)
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            if args.pretraining_flag:
                self.model_output_class = BertForPreTrainingOutput
            else:
                self.model_output_class = SequenceClassifierOutput

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
                data = send_to_device(data, torch.cuda.current_device())
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
                loss_mask = (data_b["labels"] != -100).to(torch.float)
            else:
                lm_labels = None
                loss_mask = None
            if "next_sentence_label" in data_b:
                sentence_order = data_b["next_sentence_label"].long()
            else:
                sentence_order = None

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
    def __init__(self, args):
        super().__init__("GPTTrainStep")
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = CausalLMOutputWithCrossAttentions

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
                data = {"input_ids": data["input_ids"]}
                data = send_to_device(data, torch.cuda.current_device())
            else:
                data = None
            data_b = broadcast_data(data)
            tokens_ = data_b["input_ids"].long()
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            # Get the masks and postition ids.
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, 0, False, False, False)
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
    def __init__(self, args):
        super().__init__("T5TrainStep")
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = Seq2SeqLMOutput

    @staticmethod
    def attn_mask_postprocess(attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)
        # [b, s, 1]
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        # [b, s, s]
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        # Convert attention mask to binary:
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    @staticmethod
    def get_decoder_mask(seq_length, device):
        attention_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device))
        attention_mask = attention_mask < 0.5
        return attention_mask

    @staticmethod
    def get_enc_dec_mask(attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)
        extended_attention_mask = attention_mask_b1s < 0.5
        return extended_attention_mask

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch_megatron(data_iterator):
            """Build the batch."""

            keys = ["text_enc", "text_dec", "labels", "loss_mask", "enc_mask", "dec_mask", "enc_dec_mask"]
            datatype = torch.int64

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens_enc = data_b["text_enc"].long()
            tokens_dec = data_b["text_dec"].long()
            labels = data_b["labels"].long()
            loss_mask = data_b["loss_mask"].float()

            enc_mask = data_b["enc_mask"] < 0.5
            dec_mask = data_b["dec_mask"] < 0.5
            enc_dec_mask = data_b["enc_dec_mask"] < 0.5

            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""

            # Broadcast data.
            if data_iterator is not None:
                data = next(data_iterator)
                data = send_to_device(data, torch.cuda.current_device())
            else:
                data = None
            data_b = broadcast_data(data)
            tokens_enc = data_b["input_ids"].long()
            labels = data_b["labels"].long()
            loss_mask = (labels != -100).to(torch.float)
            if "decoder_input_ids" in data_b:
                tokens_dec = data_b["decoder_input_ids"].long()
            else:
                tokens_dec = labels.new_zeros(labels.shape, device=labels.device, dtype=torch.long)
                tokens_dec[..., 1:] = labels[..., :-1].clone()
                tokens_dec[..., 0] = 0
                tokens_dec.masked_fill_(tokens_dec == -100, 0)
            enc_mask = T5TrainStep.attn_mask_postprocess(data_b["attention_mask"].long())
            dec_mask = T5TrainStep.get_decoder_mask(tokens_dec.shape[1], tokens_dec.device)
            enc_dec_mask = T5TrainStep.get_enc_dec_mask(data_b["attention_mask"].long())

            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):
        def loss_func(loss_mask, output_tensor):
            lm_loss_ = output_tensor.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

            loss = lm_loss
            averaged_losses = average_losses_across_data_parallel_group([lm_loss])

            return loss, {"lm loss": averaged_losses[0]}

        return loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            # Get the batch.
            tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = self.get_batch(
                data_iterator
            )
            # Forward model lm_labels
            output_tensor = model(
                tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=lm_labels
            )

            return output_tensor, partial(self.loss_func, loss_mask)

        return forward_step


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
