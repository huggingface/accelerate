# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os

import torch

from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, FSDP_PYTORCH_VERSION, OPTIMIZER_NAME
from .imports import is_torch_distributed_available
from .modeling import is_peft_model
from .versions import is_torch_version


if is_torch_version(">=", FSDP_PYTORCH_VERSION) and is_torch_distributed_available():
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


logger = get_logger(__name__)


def _get_model_state_dict(model, adapter_only=False):
    if adapter_only and is_peft_model(model):
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(model, adapter_name=model.active_adapter)
    else:
        return model.state_dict()


def _set_model_state_dict(model, state_dict, adapter_only=False):
    if adapter_only and is_peft_model(model):
        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(model, state_dict, adapter_name=model.active_adapter)
    else:
        return model.load_state_dict(state_dict)


def save_fsdp_model(fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False):
    os.makedirs(output_dir, exist_ok=True)

    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    with FSDP.state_dict_type(
        model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
    ):
        state_dict = _get_model_state_dict(model, adapter_only=adapter_only)
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                logger.info(f"Saving model to {output_model_file}")
                torch.save(state_dict, output_model_file)
                logger.info(f"Model saved to {output_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            output_model_file = os.path.join(output_dir, weights_name)
            logger.info(f"Saving model to {output_model_file}")
            torch.save(state_dict, output_model_file)
            logger.info(f"Model saved to {output_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{model_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Saving model to {ckpt_dir}")
            state_dict = {"model": state_dict}

            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
                planner=DefaultSavePlanner(),
            )
            logger.info(f"Model saved to {ckpt_dir}")


def load_fsdp_model(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process
    with FSDP.state_dict_type(
        model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
    ):
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if type(model) != FSDP and accelerator.process_index != 0:
                if not fsdp_plugin.sync_module_states:
                    raise ValueError(
                        "Set the `sync_module_states` flag to `True` so that model states are synced across processes when "
                        "initializing FSDP object"
                    )
                return
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = (
                os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{model_index}")
                if f"{FSDP_MODEL_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading model from {ckpt_dir}")
            state_dict = {"model": _get_model_state_dict(model, adapter_only=adapter_only)}
            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
                planner=DefaultLoadPlanner(),
            )
            state_dict = state_dict["model"]
            logger.info(f"Model loaded from {ckpt_dir}")
        load_result = _set_model_state_dict(model, state_dict, adapter_only=adapter_only)
    return load_result


def save_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0):
    os.makedirs(output_dir, exist_ok=True)
    with FSDP.state_dict_type(
        model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
    ):
        optim_state = FSDP.optim_state_dict(model, optimizer)
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if accelerator.process_index == 0:
                optim_state_name = (
                    f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
                )
                output_optimizer_file = os.path.join(output_dir, optim_state_name)
                logger.info(f"Saving Optimizer state to {output_optimizer_file}")
                torch.save(optim_state, output_optimizer_file)
                logger.info(f"Optimizer state saved in {output_optimizer_file}")
        else:
            ckpt_dir = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Saving Optimizer state to {ckpt_dir}")
            dist_cp.save_state_dict(
                state_dict={"optimizer": optim_state},
                storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
                planner=DefaultSavePlanner(),
            )
            logger.info(f"Optimizer state saved in {ckpt_dir}")


def load_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, input_dir, optimizer_index=0, adapter_only=False):
    accelerator.wait_for_everyone()
    with FSDP.state_dict_type(
        model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
    ):
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            optim_state = None
            if accelerator.process_index == 0 or not fsdp_plugin.optim_state_dict_config.rank0_only:
                optimizer_name = (
                    f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
                )
                input_optimizer_file = os.path.join(input_dir, optimizer_name)
                logger.info(f"Loading Optimizer state from {input_optimizer_file}")
                optim_state = torch.load(input_optimizer_file)
                logger.info(f"Optimizer state loaded from {input_optimizer_file}")
        else:
            ckpt_dir = (
                os.path.join(input_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
                if f"{OPTIMIZER_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading Optimizer from {ckpt_dir}")
            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=_get_model_state_dict(model, adapter_only=adapter_only),
                optimizer_key="optimizer",
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
            )
            optim_state = optim_state["optimizer"]
            logger.info(f"Optimizer loaded from {ckpt_dir}")
        flattened_osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=optim_state)
        optimizer.load_state_dict(flattened_osd)
