#!/usr/bin/env python

# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import enum
import logging
from pathlib import Path

import yaml

from accelerate.commands.utils import CustomArgumentParser


class ConversionStatus(enum.Enum):
    NOT_YET_IMPLEMENTED = 0
    REMOVED = -1


ARGUMENT_KEY_MAPPING = {
    # New keys in FSDP2
    "fsdp_version": "fsdp_version",
    "fsdp_reshard_after_forward": "fsdp_reshard_after_forward",
    # https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
    # https://huggingface.co/docs/accelerate/en/usage_guides/fsdp
    "fsdp_auto_wrap_policy": "fsdp_auto_wrap_policy",
    "fsdp_backward_prefetch": ConversionStatus.REMOVED,
    "fsdp_forward_prefetch": ConversionStatus.NOT_YET_IMPLEMENTED,
    "fsdp_cpu_ram_efficient_loading": "fsdp_cpu_ram_efficient_loading",
    "fsdp_offload_params": "fsdp_offload_params",
    "fsdp_sharding_strategy": "fsdp_reshard_after_forward",
    "fsdp_state_dict_type": "fsdp_state_dict_type",
    "fsdp_sync_module_states": ConversionStatus.REMOVED,
    "fsdp_transformer_layer_cls_to_wrap": "fsdp_transformer_layer_cls_to_wrap",
    "fsdp_min_num_params": "fsdp_min_num_params",
    "fsdp_use_orig_params": ConversionStatus.REMOVED,
    "fsdp_activation_checkpointing": "fsdp_activation_checkpointing",
}

ARGUMENT_VALUE_MAPPING = {
    "fsdp_sharding_strategy": {
        "FULL_SHARD": True,
        "SHARD_GRAD_OP": False,
        "HYBRID_SHARD": True,
        "HYBRID_SHARD_ZERO2": False,
        "NO_SHARD": False,
    },
    "fsdp_reshard_after_forward": {  # Needed to convert newly created configs using FSDP1 to FSDP2
        "FULL_SHARD": True,
        "SHARD_GRAD_OP": False,
        "HYBRID_SHARD": True,
        "HYBRID_SHARD_ZERO2": False,
        "NO_SHARD": False,
    },
}

logger = logging.getLogger(__name__)


def _validate_to_fsdp2_args(args):
    if not Path(args.config_file).exists():
        raise FileNotFoundError(f"Config file {args.config_file} not found")

    if not args.overwrite and args.output_file is None:
        raise ValueError("If --overwrite is not set, --output_file must be provided")

    if not args.overwrite and Path(args.output_file).exists():
        raise FileExistsError(f"Output file {args.output_file} already exists and --overwrite is not set")


def convert_config_to_fsdp2(config: dict) -> dict:
    fsdp_config = config.get("fsdp_config", {})

    if not fsdp_config:
        logger.info("No FSDP config found in the config file, skipping conversion...")
        return config

    new_fsdp_config = {}

    if fsdp_config.get("fsdp_version", 1) == 2:
        logger.warning("Config already specfies FSDP2, skipping conversion...")
        logger.warning(
            "If the config doesn't use new argument names, change `fsdp_version` to `1` and rerun the command."
        )
        return config

    for key, value in fsdp_config.items():
        conversion_status = ARGUMENT_KEY_MAPPING.get(key, None)
        if isinstance(conversion_status, ConversionStatus) or conversion_status is None:
            conversion_status = key
            new_fsdp_config[conversion_status] = value
            continue

        if conversion_status == ConversionStatus.REMOVED:
            logger.warning(f"Argument {key} has been removed in FSDP2, skipping this key...")
            continue

        if conversion_status == ConversionStatus.NOT_YET_IMPLEMENTED:
            logger.warning(f"Argument {key} is not yet implemented in FSDP2, skipping this key...")
            continue

        if conversion_status is None:
            logger.warning(f"Argument {key} is not being converted, skipping this key...")
            new_fsdp_config[key] = value
        else:
            if key in ARGUMENT_VALUE_MAPPING:
                value = ARGUMENT_VALUE_MAPPING[key].get(value, value)
            new_fsdp_config[ARGUMENT_KEY_MAPPING[key]] = value

    new_fsdp_config["fsdp_version"] = 2
    config["fsdp_config"] = new_fsdp_config
    return config


def to_fsdp2_command_parser(subparsers=None):
    description = "Convert an Accelerate config from FSDP1 to FSDP2"

    if subparsers is not None:
        parser = subparsers.add_parser("to-fsdp2", description=description)
    else:
        parser = CustomArgumentParser(description=description)

    parser.add_argument("--config_file", type=str, help="The config file to convert to FSDP2", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the config file if it exists",
        default=False,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to the output file to write the converted config to. If not provided, the input file will be overwritten (if --overwrite is set)",
        default=None,
    )
    if subparsers is not None:
        parser.set_defaults(func=to_fsdp2_command)

    return parser


def load_config(config_file: str) -> dict:
    with open(config_file) as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError("Config file is empty")

    return config


def to_fsdp2_command(args):
    _validate_to_fsdp2_args(args)
    config = load_config(args.config_file)

    if args.overwrite and args.output_file is None:
        args.output_file = args.config_file

    new_config = convert_config_to_fsdp2(config)

    with open(args.output_file, "w") as f:
        yaml.dump(new_config, f)
