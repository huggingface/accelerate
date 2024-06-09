from accelerate import Accelerator
from accelerate.utils import DDPCommunicationHookType


def main():
    for hook in (
        DDPCommunicationHookType.NO,
        DDPCommunicationHookType.FP16,
        DDPCommunicationHookType.BF16,
        DDPCommunicationHookType.POWER_SGD,
        DDPCommunicationHookType.BATCHED_POWER_SGD
    ):
        accelerator = Accelerator()


if __name__ == "__main__":
    main()
