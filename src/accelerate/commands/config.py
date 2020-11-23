import argparse
import json
import os
from dataclasses import dataclass

from ..config import DistributedType


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
cache_dir = os.path.join(hf_cache_home, "accelerate")
default_config_file = os.path.join(cache_dir, "default_config.json")


@dataclass
class LaunchConfig:
    distributed_type: DistributedType
    num_processes: int
    fp16: bool

    @classmethod
    def from_json_file(cls, json_file=None):
        json_file = default_config_file if json_file is None else json_file
        with open(json_file, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_json_file(self, json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            content = json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"
            f.write(content)


def config_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("config")
    else:
        parser = argparse.ArgumentParser("Accelerate config command")

    parser.add_argument(
        "--config_file",
        default=None,
        help=(
            "The path to use to store the config file. Will default to a file named default_config.json in the cache "
            "location, which is the content of the enviromnent `HF_HOME` suffixed with 'accelerate', or if you don't have "
            "such an enviromnent variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed "
            "with 'huggingface'."
        ),
    )

    if subparsers is not None:
        parser.set_defaults(func=config_command)
    return parser


def _ask_field(input_text, convert_value, default=None, error_message=None):
    ask_again = True
    while ask_again:
        result = input(input_text)
        try:
            if default is not None and len(result) == 0:
                return default
            return convert_value(result)
        except:
            if error_message is not None:
                print(error_message)
            else:
                pass


def get_user_input():
    def _convert_distributed_mode(value):
        value = int(value)
        return DistributedType(["NO", "MULTI_GPU", "TPU"][value])

    distributed_type = _ask_field(
        "Which type of machine are you using? ([0] No distributed training, [1] multi-GPU, [2] TPU): ",
        _convert_distributed_mode,
        error_message="Please enter 0, 1 or 2.",
    )

    num_processes = _ask_field(
        "How many processes will you use? [1]: ", lambda x: int(x), default=1, error_message="Please enter an integer."
    )

    def _convert_fp16(value):
        return {"yes": True, "no": False}[value.lower()]

    if distributed_type != DistributedType.TPU:
        fp16 = _ask_field(
            "Do you wish to use FP16 (mixed precision)? [yes/NO]: ",
            _convert_fp16,
            default=False,
            error_message="Please enter yes or no.",
        )

    return LaunchConfig(distributed_type=distributed_type, num_processes=num_processes, fp16=fp16)


def config_command(args):
    config = get_user_input()
    if args.config_file is not None:
        config_file = args.config_file
    else:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        config_file = default_config_file

    config.to_json_file(config_file)


def main():
    parser = config_command_parser()
    args = parser.parse_args()
    config_command(args)


if __name__ == "__main__":
    main()
