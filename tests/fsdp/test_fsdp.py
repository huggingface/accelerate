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


import functools
import os

import torch
from transformers import AutoModel

from accelerate.accelerator import Accelerator
from accelerate.state import AcceleratorState
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    TempDirTestCase,
    execute_subprocess_async,
    get_launch_command,
    path_in_accelerate_package,
    require_fp16,
    require_multi_device,
    require_non_cpu,
    require_non_torch_xla,
    run_first,
    slow,
)
from accelerate.utils import is_bf16_available, is_fp16_available, is_hpu_available, patch_environment, set_seed
from accelerate.utils.constants import (
    FSDP_AUTO_WRAP_POLICY,
    FSDP_BACKWARD_PREFETCH,
    FSDP_SHARDING_STRATEGY,
    FSDP_STATE_DICT_TYPE,
)
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from accelerate.utils.fsdp_utils import disable_fsdp_ram_efficient_loading, enable_fsdp_ram_efficient_loading


set_seed(42)


BERT_BASE_CASED = "bert-base-cased"
LLAMA_TESTING = "hf-internal-testing/tiny-random-LlamaForCausalLM"
FP16 = "fp16"
BF16 = "bf16"

dtypes = []
if is_fp16_available():
    dtypes.append(FP16)
if is_bf16_available():
    dtypes.append(BF16)


@require_non_cpu
@require_non_torch_xla
class FSDPPluginIntegration(AccelerateTestCase):
    def setUp(self):
        super().setUp()

        self.dist_env = dict(
            MASTER_ADDR="localhost",
            MASTER_PORT="10999",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

        self.fsdp_env = dict(ACCELERATE_USE_FSDP="true", **self.dist_env)

    def test_sharding_strategy(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

        # check that giving enums works fine
        for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
            env = self.fsdp_env.copy()
            env["FSDP_SHARDING_STRATEGY"] = f"{i + 1}"
            with patch_environment(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                assert fsdp_plugin.reshard_after_forward == ShardingStrategy(i + 1)
            fsdp_plugin = FullyShardedDataParallelPlugin(reshard_after_forward=ShardingStrategy(i + 1))
            assert fsdp_plugin.reshard_after_forward == ShardingStrategy(i + 1)

        # check that giving names works fine
        for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
            env = self.fsdp_env.copy()
            env["FSDP_SHARDING_STRATEGY"] = strategy
            with patch_environment(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                assert fsdp_plugin.reshard_after_forward == ShardingStrategy(i + 1)
            fsdp_plugin = FullyShardedDataParallelPlugin(reshard_after_forward=strategy)
            assert fsdp_plugin.reshard_after_forward == ShardingStrategy(i + 1)

    def test_backward_prefetch(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch

        for i, prefetch_policy in enumerate(FSDP_BACKWARD_PREFETCH):
            expected_value = None if prefetch_policy == "NO_PREFETCH" else BackwardPrefetch(i + 1)
            env = self.fsdp_env.copy()
            env["FSDP_BACKWARD_PREFETCH"] = prefetch_policy
            with patch_environment(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                assert (
                    fsdp_plugin.backward_prefetch == expected_value
                ), f"Actual: {fsdp_plugin.backward_prefetch} != Expected: {expected_value}"

            # Check if torch enum works
            if prefetch_policy != "NO_PREFETCH":
                fsdp_plugin = FullyShardedDataParallelPlugin(backward_prefetch=BackwardPrefetch(i + 1))
                assert fsdp_plugin.backward_prefetch == expected_value

            # Check if name works
            fsdp_plugin = FullyShardedDataParallelPlugin(backward_prefetch=prefetch_policy)
            assert fsdp_plugin.backward_prefetch == expected_value

    def test_state_dict_type(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        for i, state_dict_type in enumerate(FSDP_STATE_DICT_TYPE):
            env = self.fsdp_env.copy()
            env["FSDP_STATE_DICT_TYPE"] = state_dict_type
            with patch_environment(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                assert fsdp_plugin.state_dict_type == StateDictType(i + 1)
                if state_dict_type == "FULL_STATE_DICT":
                    assert fsdp_plugin.state_dict_config.offload_to_cpu
                    assert fsdp_plugin.state_dict_config.rank0_only

            fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_type=StateDictType(i + 1))
            assert fsdp_plugin.state_dict_type == StateDictType(i + 1)
            if state_dict_type == "FULL_STATE_DICT":
                assert fsdp_plugin.state_dict_config.offload_to_cpu
                assert fsdp_plugin.state_dict_config.rank0_only

        # We can also override the state_dict_type,
        # typical case: user trains with sharded, but final save is with full
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_type="FULL_STATE_DICT")
        fsdp_plugin.set_state_dict_type("SHARDED_STATE_DICT")
        assert fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT

    def test_auto_wrap_policy(self):
        for model_name in [LLAMA_TESTING, BERT_BASE_CASED]:
            model = AutoModel.from_pretrained(model_name)
            layer_to_wrap = "LlamaDecoderLayer" if model_name == LLAMA_TESTING else "BertLayer"
            for policy in FSDP_AUTO_WRAP_POLICY:
                env = self.fsdp_env.copy()
                env["FSDP_AUTO_WRAP_POLICY"] = policy
                transformer_cls_to_wrap = None
                min_num_params = None
                env.pop("FSDP_TRANSFORMER_CLS_TO_WRAP", None)
                env.pop("FSDP_MIN_NUM_PARAMS", None)
                if policy == "TRANSFORMER_BASED_WRAP":
                    env["FSDP_TRANSFORMER_CLS_TO_WRAP"] = layer_to_wrap
                    transformer_cls_to_wrap = layer_to_wrap
                elif policy == "SIZE_BASED_WRAP":
                    env["FSDP_MIN_NUM_PARAMS"] = "2000"
                    min_num_params = 2000
                # First test via env
                with patch_environment(**env):
                    fsdp_plugin = FullyShardedDataParallelPlugin()
                    fsdp_plugin.set_auto_wrap_policy(model)
                if policy == "NO_WRAP":
                    assert fsdp_plugin.auto_wrap_policy is None
                else:
                    assert isinstance(fsdp_plugin.auto_wrap_policy, functools.partial)

                # Then manually set the policy
                fsdp_plugin = FullyShardedDataParallelPlugin(
                    auto_wrap_policy=policy,
                    transformer_cls_names_to_wrap=transformer_cls_to_wrap,
                    min_num_params=min_num_params,
                )
                fsdp_plugin.set_auto_wrap_policy(model)
                if policy == "NO_WRAP":
                    assert fsdp_plugin.auto_wrap_policy is None
                else:
                    assert isinstance(fsdp_plugin.auto_wrap_policy, functools.partial)

        env = self.fsdp_env.copy()
        env["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
        env["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "T5Layer"
        with patch_environment(**env):
            fsdp_plugin = FullyShardedDataParallelPlugin()
            with self.assertRaises(Exception) as cm:
                fsdp_plugin.set_auto_wrap_policy(model)
            assert "Could not find the transformer layer class T5Layer in the model." in str(cm.exception)

        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap="T5Layer",
        )
        with self.assertRaises(Exception) as cm:
            fsdp_plugin.set_auto_wrap_policy(model)
        assert "Could not find the transformer layer class T5Layer in the model." in str(cm.exception)

        env = self.fsdp_env.copy()
        env["FSDP_AUTO_WRAP_POLICY"] = "SIZE_BASED_WRAP"
        env["FSDP_MIN_NUM_PARAMS"] = "0"
        with patch_environment(**env):
            fsdp_plugin = FullyShardedDataParallelPlugin()
            fsdp_plugin.set_auto_wrap_policy(model)
            assert fsdp_plugin.auto_wrap_policy is None

        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy="SIZE_BASED_WRAP",
            min_num_params=0,
        )
        fsdp_plugin.set_auto_wrap_policy(model)
        assert fsdp_plugin.auto_wrap_policy is None

    def test_mixed_precision(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        for mp_dtype in dtypes:
            env = self.fsdp_env.copy()
            env["ACCELERATE_MIXED_PRECISION"] = mp_dtype
            with patch_environment(**env):
                accelerator = Accelerator()
                if mp_dtype == "fp16":
                    dtype = torch.float16
                elif mp_dtype == "bf16":
                    dtype = torch.bfloat16
                mp_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
                assert accelerator.state.fsdp_plugin.mixed_precision_policy == mp_policy
                if mp_dtype == FP16:
                    assert isinstance(accelerator.scaler, ShardedGradScaler)
                elif mp_dtype == BF16:
                    assert accelerator.scaler is None
                AcceleratorState._reset_state(True)
            plugin = FullyShardedDataParallelPlugin(
                mixed_precision_policy={"param_dtype": dtype, "reduce_dtype": dtype, "buffer_dtype": dtype}
            )
            assert plugin.mixed_precision_policy == mp_policy
            with patch_environment(**self.dist_env):
                accelerator = Accelerator(fsdp_plugin=plugin)
                assert accelerator.state.fsdp_plugin.mixed_precision_policy == mp_policy
            AcceleratorState._reset_state(True)

    def test_mixed_precision_buffer_autocast_override(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        for mp_dtype in dtypes:
            if mp_dtype == "fp16":
                dtype = torch.float16
            elif mp_dtype == "bf16":
                dtype = torch.bfloat16
            mp_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=torch.float32)

            env = self.fsdp_env.copy()
            env["ACCELERATE_MIXED_PRECISION"] = mp_dtype
            with patch_environment(**env):
                accelerator = Accelerator()
                accelerator.state.fsdp_plugin.set_mixed_precision(dtype, buffer_autocast=True, override=True)
                assert accelerator.state.fsdp_plugin.mixed_precision_policy == mp_policy
                if mp_dtype == FP16:
                    assert isinstance(accelerator.scaler, ShardedGradScaler)
                elif mp_dtype == BF16:
                    assert accelerator.scaler is None
                AcceleratorState._reset_state(True)

    def test_cpu_offload(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

        for flag in [True, False]:
            env = self.fsdp_env.copy()
            env["FSDP_OFFLOAD_PARAMS"] = str(flag).lower()
            with patch_environment(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                assert fsdp_plugin.cpu_offload == CPUOffload(offload_params=flag)

            fsdp_plugin = FullyShardedDataParallelPlugin(cpu_offload=flag)
            assert fsdp_plugin.cpu_offload == CPUOffload(offload_params=flag)

    def test_cpu_ram_efficient_loading(self):
        enable_fsdp_ram_efficient_loading()
        fsdp_plugin = FullyShardedDataParallelPlugin()
        assert fsdp_plugin.cpu_ram_efficient_loading is True
        assert os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING") == "True"
        disable_fsdp_ram_efficient_loading()
        fsdp_plugin = FullyShardedDataParallelPlugin()
        assert fsdp_plugin.cpu_ram_efficient_loading is False
        assert os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING") == "False"


@run_first
# Skip this test when TorchXLA is available because accelerate.launch does not support TorchXLA FSDP.
@require_non_torch_xla
@require_multi_device
@slow
class FSDPIntegrationTest(TempDirTestCase):
    test_scripts_folder = path_in_accelerate_package("test_utils", "scripts", "external_deps")

    def setUp(self):
        super().setUp()
        self.performance_lower_bound = 0.70 if is_hpu_available() else 0.82
        self.performance_configs = [
            "fsdp_shard_grad_op_transformer_based_wrap",
            "fsdp_full_shard_transformer_based_wrap",
        ]
        self.peak_memory_usage_upper_bound = {
            "multi_gpu_fp16": 3200,
            "fsdp_shard_grad_op_transformer_based_wrap_fp16": 2000,
            "fsdp_full_shard_transformer_based_wrap_fp16": 1900,
            # Disabling below test as it overwhelms the RAM memory usage
            # on CI self-hosted runner leading to tests getting killed.
            # "fsdp_full_shard_cpu_offload_transformer_based_wrap_fp32": 1500,  # fp16 was leading to indefinite hang
        }
        self.n_train = 160
        self.n_val = 160

    @require_fp16
    def test_performance(self):
        self.test_file_path = self.test_scripts_folder / "test_performance.py"
        cmd = get_launch_command(num_processes=2, num_machines=1, machine_rank=0, use_fsdp=True)
        for config in self.performance_configs:
            cmd_config = cmd.copy()
            for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
                if strategy.lower() in config:
                    cmd_config.append(f"--fsdp_reshard_after_forward={strategy}")
                    break

            if "fp32" in config:
                cmd_config.append("--mixed_precision=no")
            else:
                cmd_config.append("--mixed_precision=fp16")

            if "cpu_offload" in config:
                cmd_config.append("--fsdp_offload_params=True")

            for policy in FSDP_AUTO_WRAP_POLICY:
                if policy.lower() in config:
                    cmd_config.append(f"--fsdp_auto_wrap_policy={policy}")
                    break

            if policy == "TRANSFORMER_BASED_WRAP":
                cmd_config.append("--fsdp_transformer_layer_cls_to_wrap=BertLayer")
            elif policy == "SIZE_BASED_WRAP":
                cmd_config.append("--fsdp_min_num_params=2000")

            cmd_config.extend(
                [
                    self.test_file_path,
                    f"--output_dir={self.tmpdir}",
                    f"--performance_lower_bound={self.performance_lower_bound}",
                ]
            )

            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd_config)

    @require_fp16
    def test_checkpointing(self):
        self.test_file_path = self.test_scripts_folder / "test_checkpointing.py"
        cmd = get_launch_command(
            num_processes=2,
            num_machines=1,
            machine_rank=0,
            use_fsdp=True,
            mixed_precision="fp16",
            fsdp_transformer_layer_cls_to_wrap="BertLayer",
        )

        for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
            cmd_config = cmd.copy()
            cmd_config.append(f"--fsdp_reshard_after_forward={strategy}")
            if strategy != "FULL_SHARD":
                continue
            state_dict_config_index = len(cmd_config)
            for state_dict_type in FSDP_STATE_DICT_TYPE:
                # Todo: Currently failing for `LOCAL_STATE_DICT` with error
                # Unexpected key(s) in state_dict: "_fsdp_wrapped_module._flat_param".
                if state_dict_type == "LOCAL_STATE_DICT":
                    continue

                cmd_config = cmd_config[:state_dict_config_index]
                cmd_config.append(f"--fsdp_state_dict_type={state_dict_type}")
                cmd_config.extend(
                    [
                        self.test_file_path,
                        f"--output_dir={self.tmpdir}",
                        "--partial_train_epoch=1",
                    ]
                )
                with patch_environment(omp_num_threads=1):
                    execute_subprocess_async(cmd_config)

                cmd_config = cmd_config[:-1]
                resume_from_checkpoint = os.path.join(self.tmpdir, "epoch_0")
                cmd_config.extend(
                    [
                        f"--resume_from_checkpoint={resume_from_checkpoint}",
                    ]
                )
                with patch_environment(omp_num_threads=1):
                    execute_subprocess_async(cmd_config)

    @require_fp16
    def test_peak_memory_usage(self):
        self.test_file_path = self.test_scripts_folder / "test_peak_memory_usage.py"
        cmd = get_launch_command(num_processes=2, num_machines=1, machine_rank=0)
        for spec, peak_mem_upper_bound in self.peak_memory_usage_upper_bound.items():
            cmd_config = cmd.copy()
            if "fp16" in spec:
                cmd_config.extend(["--mixed_precision=fp16"])
            else:
                cmd_config.extend(["--mixed_precision=no"])

            if "multi_gpu" in spec:
                continue
            else:
                cmd_config.extend(["--use_fsdp"])
                for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
                    if strategy.lower() in spec:
                        cmd_config.append(f"--fsdp_reshard_after_forward={strategy}")
                        break

                if "cpu_offload" in spec:
                    cmd_config.append("--fsdp_offload_params=True")

                for policy in FSDP_AUTO_WRAP_POLICY:
                    if policy.lower() in spec:
                        cmd_config.append(f"--fsdp_auto_wrap_policy={policy}")
                        break

                if policy == "TRANSFORMER_BASED_WRAP":
                    cmd_config.append("--fsdp_transformer_layer_cls_to_wrap=BertLayer")
                elif policy == "SIZE_BASED_WRAP":
                    cmd_config.append("--fsdp_min_num_params=2000")

            cmd_config.extend(
                [
                    self.test_file_path,
                    f"--output_dir={self.tmpdir}",
                    f"--peak_memory_upper_bound={peak_mem_upper_bound}",
                    f"--n_train={self.n_train}",
                    f"--n_val={self.n_val}",
                ]
            )
            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd_config)
