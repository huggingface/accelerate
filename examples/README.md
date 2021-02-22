# In this folder we showcase various full examples using `Accelerate`

## Simple NLP example

The [simple_example.py](./simple_example.py) script is a simple example to train a Bert model on a classification task ([GLUE's MRPC]()).

The same script can be run in any of the following configurations:
- single CPU or single GPU
- multi GPUS (using PyTorch distributed mode)
- (multi) TPUs
- fp16 (mixed-precision) or fp32 (normal precision)

To run it in each of these various modes, use the following commands:
- single CPU:
    * from a server without GPU
        ```bash
        python ./simple_example.py
        ```
    * from any server
        ```bash
        python ./simple_example.py --device cpu
        ```
- single GPU:
    ```bash
    python ./simple_example.py  # from a server with a GPU
    ```
- with fp16 (mixed-precision)
    ```bash
    python ./simple_example.py --fp16
    ```
- multi GPUS (using PyTorch distributed mode)
    * With Accelerate config and launcher
        ```bash
        accelerate config  # This will create a config file on each server
        accelerate launch ./simple_example.py  # This will run the script on each server
        ```
    * With PyTorch launcher and Accelerate config
        ```bash
        accelerate config  # This will create a local ENV file on each server
        python -m torch.distributed.launch --nproc_per_node 2 --use_env ./simple_example.py
        ```
    * With PyTorch launcher only
        ```bash
        python -m torch.distributed.launch --nproc_per_node 2 --local_rank 0 ./simple_example.py  # On the first server
        python -m torch.distributed.launch --nproc_per_node 2 --local_rank 1 ./simple_example.py  # On the second server
        ```
- (multi) TPUs
    * With Accelerate config and launcher
        ```bash
        accelerate config  # This will create a config file on each server
        accelerate launch ./simple_example.py  # This will run the script on each server
        ```
    * With PyTorch launcher and Accelerate config
        ```bash
        accelerate config  # This will create a local ENV file on each server
        python -m torch.distributed.launch --nproc_per_node 2 --use_env ./simple_example.py
        ```
    * With PyTorch launcher only
        ```bash
        python -m torch.distributed.launch --nproc_per_node 2 --local_rank 0 ./simple_example.py  # On the first server
        python -m torch.distributed.launch --nproc_per_node 2 --local_rank 1 ./simple_example.py  # On the second server
        ```
