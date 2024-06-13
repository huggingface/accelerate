#!/bin/bash -l

#SBATCH --job-name=multicpu
#SBATCH --nodes=2                       # number of Nodes
#SBATCH --ntasks-per-node=1             # number of MP tasks
#SBATCH --exclusive
#SBATCH --output=output/torch_%A.log
#SBATCH --cpus-per-task=56

######################
### Set enviroment ###
######################
source activateEnviroment.sh

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

# Setup env variables for distributed jobs
MASTER_PORT="${MASTER_PORT:-29555 }"
echo "head_node_ip=${head_node_ip}"
echo "MASTER_PORT=${MASTER_PORT}"


INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-1}"

if [[ $SLURM_NNODES == 1 ]] && [[ $INSTANCES_PER_NODE == 1 ]]; then
  export CCL_WORKER_COUNT=0
  LAUNCHER=""
else
  # Setup env variables for distributed jobs
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_PORT="${MASTER_PORT:-25679}"
  export CCL_WORKER_COUNT="${CCL_WORKER_COUNT:-2}"
  export OMP_NUM_THREADS=56
  
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

  # Write hostfile
  HOSTFILE_PATH=hostfile
  scontrol show hostname $SLURM_JOB_NODELIST | perl -ne 'chomb; print "$_"x1'> ${HOSTFILE_PATH}

  export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * ${INSTANCES_PER_NODE})) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --num_cpu_threads_per_process $OMP_NUM_THREADS \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mpirun_hostfile $HOSTFILE_PATH \
    --mpirun_ccl $CCL_WORKER_COUNT"
fi

# This step is necessary because accelerate launch does not handle multiline arguments properly
export SCRIPT="/accelerate/examples/complete_nlp_example.py"
export SCRIPT_ARGS=" \
    --cpu \
    --output_dir /accelerate/examples/output \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
# Print the command
echo $CMD
echo ""

# Run the command
eval $CMD