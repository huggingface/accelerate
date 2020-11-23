#!/usr/bin/env python
from argparse import ArgumentParser

from .config import config_command_parser
from .launch import launch_command_parser


def main():
    parser = ArgumentParser("Accelerate CLI tool", usage="accelerate <command> [<args>]")
    subparsers = parser.add_subparsers(help="accelerate command helpers")

    # Register commands
    config_command_parser(subparsers=subparsers)
    launch_command_parser(subparsers=subparsers)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    args.func(args)


if __name__ == "__main__":
    main()
