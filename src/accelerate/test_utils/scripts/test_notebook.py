# Test file to ensure that in general certain situational setups for notebooks work.
import argparse

from accelerate import PartialState, notebook_launcher


parser = argparse.ArgumentParser()
parser.add_argument("--num_processes", type=int, default=1)
args = parser.parse_args()


def function():
    print(f"PartialState:\n{PartialState()}")


if __name__ == "__main__":
    notebook_launcher(function, num_processes=int(args.num_processes))
