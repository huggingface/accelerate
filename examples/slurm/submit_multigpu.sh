#!/bin/bash

#SBATCH --job-name=multigpu
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=160         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)

######################
### Set environment ###
######################
source activateEnvironment.sh
export GPUS_PER_NODE=4
######################

export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export SCRIPT="${ACCELERATE_DIR}/examples/complete_nlp_example.py"
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --output_dir ${ACCELERATE_DIR}/examples/output \
    --with_tracking \
    "

accelerate launch --num_processes $GPUS_PER_NODE $SCRIPT $SCRIPT_ARGS