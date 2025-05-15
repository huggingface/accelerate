# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import unittest

from accelerate.utils.launch import prepare_multi_gpu_env


class TestPrepareMultiGpuEnv(unittest.TestCase):
    def test_auto_port_selection(self):
        args = argparse.Namespace(
            num_processes=1,
            num_machines=1,
            main_process_ip="127.0.0.1",
            main_process_port=0,
            machine_rank=0,
            module=False,
            no_python=False,
            debug=False,
            gpu_ids="all",
            mixed_precision="no",
            dynamo_backend="NO",
            dynamo_mode="default",
            dynamo_use_fullgraph=False,
            dynamo_use_dynamic=False,
            dynamo_use_regional_compilation=False,
            use_fsdp=False,
            fsdp_cpu_ram_efficient_loading=False,
            fsdp_sync_module_states=False,
            fsdp_version=None,
            fsdp_sharding_strategy=None,
            fsdp_reshard_after_forward=False,
            fsdp_offload_params=False,
            fsdp_min_num_params=0,
            fsdp_auto_wrap_policy=None,
            fsdp_transformer_layer_cls_to_wrap=None,
            fsdp_backward_prefetch=None,
            fsdp_state_dict_type=None,
            fsdp_forward_prefetch=False,
            fsdp_use_orig_params=False,
            fsdp_activation_checkpointing=False,
            use_tp=False,
            tp_size=1,
            use_megatron_lm=False,
            megatron_lm_tp_degree=1,
            megatron_lm_pp_degree=1,
            megatron_lm_gradient_clipping=1.0,
            megatron_lm_num_micro_batches=None,
            megatron_lm_sequence_parallelism=None,
            megatron_lm_recompute_activations=None,
            megatron_lm_use_distributed_optimizer=None,
            num_cpu_threads_per_process=1,
            enable_cpu_affinity=False,
            same_network=False,
        )

        prepare_multi_gpu_env(args)
        self.assertIn("master_port", args.__dict__)
        self.assertNotEqual(args.master_port, "0")
        self.assertTrue(args.master_port.isdigit())
