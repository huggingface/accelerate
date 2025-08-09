#!/bin/bash -l

#SBATCH --job-name=multicpu
#SBATCH --nodes=2                       # number of Nodes
#SBATCH --ntasks-per-node=1             # number of MP tasks
#SBATCH --exclusive
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j

######################
### Set environment ###
######################
source activateEnvironment.sh

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

# Setup env variables for distributed jobs
export MASTER_PORT="${MASTER_PORT:-29555 }"
echo "head_node_ip=${head_node_ip}"
echo "MASTER_PORT=${MASTER_PORT}"

INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-1}"

if [[ $SLURM_NNODES == 1 ]] && [[ $INSTANCES_PER_NODE == 1 ]]; then
  export CCL_WORKER_COUNT=0
  LAUNCHER=""
else
  # Setup env variables for distributed jobs
  export CCL_WORKER_COUNT="${CCL_WORKER_COUNT:-2}"  
  echo "CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"

  # Write hostfile
  HOSTFILE_PATH=hostfile
  scontrol show hostname $SLURM_JOB_NODELIST | perl -ne 'chomb; print "$_"x1'> ${HOSTFILE_PATH}

  export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * ${INSTANCES_PER_NODE})) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port $MASTER_PORT \
    --mpirun_hostfile $HOSTFILE_PATH \
    --mpirun_ccl $CCL_WORKER_COUNT"
fi

# This step is necessary because accelerate launch does not handle multiline arguments properly
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export SCRIPT="${ACCELERATE_DIR}/examples/complete_nlp_example.py"
export SCRIPT_ARGS=" \
    --cpu \
    --output_dir ${ACCELERATE_DIR}/examples/output \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
# Print the command
echo $CMD
echo ""

# Run the command
eval $CMD