# Test file to ensure that in general certain situational setups for notebooks work.
import os

from pytest import raises

from accelerate import PartialState, notebook_launcher
from accelerate.test_utils import require_bnb
from accelerate.utils import is_bnb_available


def basic_function():
    # Just prints the PartialState
    print(f"PartialState:\n{PartialState()}")


NUM_PROCESSES = int(os.environ.get("ACCELERATE_NUM_PROCESSES", 1))


def test_can_initialize():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


@require_bnb
def test_problematic_imports():
    with raises(RuntimeError, match="Please keep these imports"):
        import bitsandbytes as bnb  # noqa: F401

        notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


def main():
    print("Test basic notebook can be ran")
    test_can_initialize()
    if is_bnb_available():
        print("Test problematic imports (bnb)")
        test_problematic_imports()


if __name__ == "__main__":
    main()
