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


import inspect
import os

import torch
from transformers import AutoModel
from transformers.testing_utils import mockenv_context
from transformers.trainer_utils import set_seed

import accelerate
from accelerate.accelerator import Accelerator
from accelerate.state import AcceleratorState
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    TempDirTestCase,
    execute_subprocess_async,
    require_fsdp,
    require_multi_device,
    require_non_cpu,
    slow,
)
from accelerate.utils.constants import (
    FSDP_AUTO_WRAP_POLICY,
    FSDP_BACKWARD_PREFETCH,
    FSDP_SHARDING_STRATEGY,
    FSDP_STATE_DICT_TYPE,
)
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from accelerate.utils.other import patch_environment


set_seed(42)

BERT_BASE_CASED = "bert-base-cased"
FP16 = "fp16"
BF16 = "bf16"
dtypes = [FP16, BF16]


@require_fsdp
@require_non_cpu
class FSDPPluginIntegration(AccelerateTestCase):
    def setUp(self):
        super().setUp()

        self.dist_env = dict(
            ACCELERATE_USE_FSDP="true",
            MASTER_ADDR="localhost",
            MASTER_PORT="10999",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

    def test_sharding_strategy(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

        for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
            env = self.dist_env.copy()
            env["FSDP_SHARDING_STRATEGY"] = f"{i + 1}"
            env["FSDP_SHARDING_STRATEGY_NAME"] = strategy
            with mockenv_context(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                self.assertEqual(fsdp_plugin.sharding_strategy, ShardingStrategy(i + 1))

    def test_backward_prefetch(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch

        for i, prefetch_policy in enumerate(FSDP_BACKWARD_PREFETCH):
            env = self.dist_env.copy()
            env["FSDP_BACKWARD_PREFETCH"] = prefetch_policy
            with mockenv_context(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                if prefetch_policy == "NO_PREFETCH":
                    self.assertIsNone(fsdp_plugin.backward_prefetch)
                else:
                    self.assertEqual(fsdp_plugin.backward_prefetch, BackwardPrefetch(i + 1))

    def test_state_dict_type(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        for i, state_dict_type in enumerate(FSDP_STATE_DICT_TYPE):
            env = self.dist_env.copy()
            env["FSDP_STATE_DICT_TYPE"] = state_dict_type
            with mockenv_context(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                self.assertEqual(fsdp_plugin.state_dict_type, StateDictType(i + 1))
                if state_dict_type == "FULL_STATE_DICT":
                    self.assertTrue(fsdp_plugin.state_dict_config.offload_to_cpu)
                    self.assertTrue(fsdp_plugin.state_dict_config.rank0_only)

    def test_auto_wrap_policy(self):
        model = AutoModel.from_pretrained(BERT_BASE_CASED)
        for policy in FSDP_AUTO_WRAP_POLICY:
            env = self.dist_env.copy()
            env["FSDP_AUTO_WRAP_POLICY"] = policy
            if policy == "TRANSFORMER_BASED_WRAP":
                env["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "BertLayer"
            elif policy == "SIZE_BASED_WRAP":
                env["FSDP_MIN_NUM_PARAMS"] = "2000"
            with mockenv_context(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                fsdp_plugin.set_auto_wrap_policy(model)
                if policy == "NO_WRAP":
                    self.assertIsNone(fsdp_plugin.auto_wrap_policy)
                else:
                    self.assertIsNotNone(fsdp_plugin.auto_wrap_policy)

        env = self.dist_env.copy()
        env["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
        env["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "T5Layer"
        with mockenv_context(**env):
            fsdp_plugin = FullyShardedDataParallelPlugin()
            with self.assertRaises(Exception) as cm:
                fsdp_plugin.set_auto_wrap_policy(model)
            self.assertTrue("Could not find the transformer layer class to wrap in the model." in str(cm.exception))

        env = self.dist_env.copy()
        env["FSDP_AUTO_WRAP_POLICY"] = "SIZE_BASED_WRAP"
        env["FSDP_MIN_NUM_PARAMS"] = "0"
        with mockenv_context(**env):
            fsdp_plugin = FullyShardedDataParallelPlugin()
            fsdp_plugin.set_auto_wrap_policy(model)
            self.assertIsNone(fsdp_plugin.auto_wrap_policy)

    def test_mixed_precision(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        for mp_dtype in dtypes:
            env = self.dist_env.copy()
            env["ACCELERATE_MIXED_PRECISION"] = mp_dtype
            with mockenv_context(**env):
                accelerator = Accelerator()
                if mp_dtype == "fp16":
                    dtype = torch.float16
                elif mp_dtype == "bf16":
                    dtype = torch.bfloat16
                mp_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
                self.assertEqual(accelerator.state.fsdp_plugin.mixed_precision_policy, mp_policy)
                if mp_dtype == FP16:
                    self.assertTrue(isinstance(accelerator.scaler, ShardedGradScaler))
                elif mp_dtype == BF16:
                    self.assertIsNone(accelerator.scaler)
                AcceleratorState._reset_state(True)

    def test_cpu_offload(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

        for flag in [True, False]:
            env = self.dist_env.copy()
            env["FSDP_OFFLOAD_PARAMS"] = str(flag).lower()
            with mockenv_context(**env):
                fsdp_plugin = FullyShardedDataParallelPlugin()
                self.assertEqual(fsdp_plugin.cpu_offload, CPUOffload(offload_params=flag))


@require_fsdp
@require_multi_device
@slow
class FSDPIntegrationTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.performance_lower_bound = 0.82
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

        mod_file = inspect.getfile(accelerate.test_utils)
        self.test_scripts_folder = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["scripts", "external_deps"])

    def test_performance(self):
        self.test_file_path = os.path.join(self.test_scripts_folder, "test_performance.py")
        cmd = ["accelerate", "launch", "--num_processes=2", "--num_machines=1", "--machine_rank=0", "--use_fsdp"]
        for config in self.performance_configs:
            cmd_config = cmd.copy()
            for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
                if strategy.lower() in config:
                    cmd_config.append(f"--fsdp_sharding_strategy={i+1}")
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
                execute_subprocess_async(cmd_config, env=os.environ.copy())

    def test_checkpointing(self):
        self.test_file_path = os.path.join(self.test_scripts_folder, "test_checkpointing.py")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
            "--use_fsdp",
            "--mixed_precision=fp16",
            "--fsdp_transformer_layer_cls_to_wrap=BertLayer",
        ]

        for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
            cmd_config = cmd.copy()
            cmd_config.append(f"--fsdp_sharding_strategy={i+1}")
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
                    execute_subprocess_async(cmd_config, env=os.environ.copy())

                cmd_config = cmd_config[:-1]
                resume_from_checkpoint = os.path.join(self.tmpdir, "epoch_0")
                cmd_config.extend(
                    [
                        f"--resume_from_checkpoint={resume_from_checkpoint}",
                    ]
                )
                with patch_environment(omp_num_threads=1):
                    execute_subprocess_async(cmd_config, env=os.environ.copy())

    def test_peak_memory_usage(self):
        self.test_file_path = os.path.join(self.test_scripts_folder, "test_peak_memory_usage.py")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
        ]
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
                        cmd_config.append(f"--fsdp_sharding_strategy={i+1}")
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
                execute_subprocess_async(cmd_config, env=os.environ.copy())
