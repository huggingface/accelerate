import argparse
import platform

import numpy as np

from accelerate import __version__ as version
from accelerate.file_utils import is_torch_available


def env_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("env")
    else:
        parser = argparse.ArgumentParser("Accelerate env command")

    if subparsers is not None:
        parser.set_defaults(func=env_command)
    return parser


def env_command(args):
    pt_version = "not installed"
    pt_cuda_available = "NA"
    if is_torch_available():
        import torch

        pt_version = torch.__version__
        pt_cuda_available = torch.cuda.is_available()

    info = {
        "`accelerate` version": version,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Numpy version": np.__version__,
        "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
    }

    print("\nCopy-and-paste the text below in your GitHub issue\n")
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]) + "\n")

    return info


def main():
    parser = env_command_parser()
    args = parser.parse_args()
    env_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
