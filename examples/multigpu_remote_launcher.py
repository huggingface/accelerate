import argparse

import runhouse as rh
import torch
from nlp_example import training_function

from accelerate.utils import PrepareForLaunch, patch_environment


def launch_train(*args):
    num_processes = torch.cuda.device_count()
    print(f"Device count: {num_processes}")
    with patch_environment(
        world_size=num_processes, master_addr="127.0.01", master_port="29500", mixed_precision=args[1].mixed_precision
    ):
        launcher = PrepareForLaunch(training_function, distributed_type="MULTI_GPU")
        torch.multiprocessing.start_processes(launcher, args=args, nprocs=num_processes, start_method="spawn")


if __name__ == "__main__":
    # Refer to https://runhouse-docs.readthedocs-hosted.com/en/main/rh_primitives/cluster.html#hardware-setup
    # for cloud access setup instructions (if using on-demand hardware), and for API specifications.

    # on-demand GPU
    # gpu = rh.cluster(name='rh-cluster', instance_type='V100:1', provider='cheapest', use_spot=False)  # single GPU
    gpu = rh.cluster(name="rh-cluster", instance_type="V100:4", provider="cheapest", use_spot=False)  # multi GPU
    gpu.up_if_not()

    # on-prem GPU
    # gpu = rh.cluster(
    #           ips=["ip_addr"], ssh_creds={ssh_user:"<username>", ssh_private_key:"<key_path>"}, name="rh-cluster"
    #       )

    # Set up remote function
    reqs = [
        "pip:./",
        "transformers",
        "datasets",
        "evaluate",
        "tqdm",
        "scipy",
        "scikit-learn",
        "tensorboard",
        "torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117",
    ]
    launch_train_gpu = rh.function(fn=launch_train, system=gpu, reqs=reqs, name="train_bert_glue")

    # Define train args/config, run train function
    train_args = argparse.Namespace(cpu=False, mixed_precision="fp16")
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    launch_train_gpu(config, train_args, stream_logs=True)

    # Alternatively, we can just run as instructed in the README (but only because there's already a wrapper CLI):
    # gpu.install_packages(reqs)
    # gpu.run(['accelerate launch --multi_gpu accelerate/examples/nlp_example.py'])
