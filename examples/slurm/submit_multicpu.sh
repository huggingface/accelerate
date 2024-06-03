#!/bin/bash -l

#SBATCH --job-name=multicpu
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=160         # number of cores per task
#SBATCH --job-name=multicpu_slurm_accelerate
#SBATCH --nodes=2
#SBATCH --output=output/torch_%A.log


######################
### Set enviroment ###
######################
source activateEnviroment.sh
export CPUS_PER_NODE=16

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

# Setup env variables for distributed jobs
MASTER_PORT="${MASTER_PORT:-29555 }"
echo "head_node_ip=${head_node_ip}"
echo "MASTER_PORT=${MASTER_PORT}"

# Write hostfile
HOSTFILE_PATH=hostfile
scontrol show hostname $SLURM_JOB_NODELIST | perl -ne 'chomb; print "$_"x1'> ${HOSTFILE_PATH}

 export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * CPUS_PER_NODE)) 
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port $MASTER_PORT \
   "

export SCRIPT="complete_nlp_example.py"
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --cpu \
    --output_dir ~/output \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS ${CMD}" 

# Print the command
echo $CMD
echo ""

srun $CMD
