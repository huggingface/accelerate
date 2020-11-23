#!/usr/bin/env python
from argparse import ArgumentParser

from .launch import LaunchCommand

def main():
    parser = ArgumentParser("Accelerate CLI tool", usage="accelerate <command> [<args>]")
    commands_parser = parser.add_subparsers(help="accelerate command helpers")

    # Register commands
    LaunchCommand.register_subcommand(parser=commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()

if __name__ == "__main__":
    main()