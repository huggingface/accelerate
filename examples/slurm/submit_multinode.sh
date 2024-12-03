#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=80         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)

######################
### Set enviroment ###
######################
source activateEnvironment.sh
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
NODE_RANK=$SLURM_PROCID
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_JOB_NUM_NODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --role $SLURMD_NODENAME: \
    --machine_rank $NODE_RANK \
    --multi_gpu \
    "
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export SCRIPT="${ACCELERATE_DIR}/examples/complete_nlp_example.py"
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --output_dir ${ACCELERATE_DIR}/examples/output \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun $CMD
