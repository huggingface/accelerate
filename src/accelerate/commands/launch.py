#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path

from . import BaseAccelerateCommand


def launch_command_factory(args: argparse.Namespace):
    return LaunchCommand(args)


class LaunchCommand(BaseAccelerateCommand):
    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        launch_parser = parser.add_parser("launch")
        launch_parser.add_argument("--multi_gpu", default=False, action="store_true", help="Whether or not this should launch a distributed GPU training.")
        launch_parser.add_argument("--tpu", default=False, action="store_true", help="Whether or not this should launch a TPU training.")
        launch_parser.add_argument("--num_processes", type=int, default=1, help="The number of processes to be launched in parallel.")
        launch_parser.add_argument("training_script", type=str, help=(
            "The full path to the script to be launched in parallel, followed by all the arguments for the training "
            "script."
        ))
        # Other arguments of the training scripts
        launch_parser.add_argument('training_script_args', nargs=argparse.REMAINDER, help="Arguments of the training script.")
        launch_parser.set_defaults(func=launch_command_factory)
    
    def __init__(self, args):
        self.args = args
        # Sanity checks
        if args.multi_gpu and args.tpu:
            raise ValueError("You can only pick one between `--multi_gpu` and `--tpu`.")

    def simple_launcher(self):
        cmd = [sys.executable, self.args.training_script]
        cmd.extend(self.args.training_script_args)

        process = subprocess.Popen(cmd)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

    def multi_gpu_launcher(self):
        cmd = [sys.executable, "-m", "torch.distributed.launch"]
        cmd.extend(["--nproc_per_node", str(self.args.num_processes)])
        cmd.append(self.args.training_script)
        cmd.extend(self.args.training_script_args)

        process = subprocess.Popen(cmd)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

    def tpu_launcher(self):
        import torch_xla.distributed.xla_multiprocessing as xmp
        # Import training_script as a module.
        script_fpath = Path(self.args.training_script)
        sys.path.append(str(script_fpath.parent.resolve()))
        mod_name = script_fpath.stem
        mod = importlib.import_module(mod_name)

        # Patch sys.argv
        sys.argv = [self.args.training_script] + self.args.training_script_args + ["--tpu_num_cores", str(self.args.num_processes)]

        xmp.spawn(mod._mp_fn, args=(), nprocs=self.args.num_processes)

    def run(self):
        if self.args.multi_gpu:
            self.multi_gpu_launcher()
        elif self.args.tpu:
            self.tpu_launcher()
        else:
            self.simple_launcher()
