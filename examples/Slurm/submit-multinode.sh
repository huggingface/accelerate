#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=160         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)

######################
### Set enviroment ###
######################
source activateEnviroment.sh
######################

######################
#### Set network #####
######################
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
######################

export LAUNCHER=" \
    torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_GPUS \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
"

export SCRIPT=/accelerate/examples/complete_nlp_example.py
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --output_dir /accelerate/examples/output \
    "

srun $LAUNCHER $SCRIPT $SCRIPT_ARGS