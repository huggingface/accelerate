#!/usr/bin/env python

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import merge_fsdp_weights


description = """Utility to merge the weights from multiple FSDP checkpoints into a single combined checkpoint. Should be used if
`SHARDED_STATE_DICT` was used for the model. Weights will be saved to `{output_path}`.

This is a CPU-bound process and requires enough RAM to load the entire model state dict."""


def merge_command(args):
    merge_fsdp_weights(
        args.checkpoint_directory, args.output_path, not args.unsafe_serialization, args.remove_checkpoint_dir
    )


def merge_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("merge-weights", description=description)
    else:
        parser = CustomArgumentParser(description=description)

    parser.add_argument("checkpoint_directory", type=str, help="A directory containing sharded weights saved by FSDP.")
    parser.add_argument(
        "output_path",
        type=str,
        help="The path to save the merged weights. Defaults to the current directory. ",
    )
    parser.add_argument(
        "--unsafe_serialization",
        action="store_false",
        default=False,
        help="Whether to save the merged weights as `.bin` rather than `.safetensors` (not recommended).",
    )
    parser.add_argument(
        "--remove_checkpoint_dir",
        action="store_true",
        help="Whether to remove the checkpoint directory after merging.",
        default=False,
    )

    if subparsers is not None:
        parser.set_defaults(func=merge_command)
    return parser


def main():
    parser = merge_command_parser()
    args = parser.parse_args()
    merge_command(args)


if __name__ == "__main__":
    main()
