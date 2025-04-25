#!/bin/bash
#SBATCH --job-name=poco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00           # Max wall time
#SBATCH --output=/path/to/fitting/pipeline/logs/%j_%x.out
#SBATCH --error=/path/to/fitting/pipeline/logs/%j_%x.err

# Critical: set threading variables (should help to use as much as possible of the CPUs available) 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check if a config file was provided
if [ -z "$1" ]; then
  echo "Error: No config file provided."
  echo "Usage: sbatch script_name.sh <config_file>"
  exit 1
fi

CONFIG_FILE=$1

module load python/conda3-2023.09
eval "$(/cosma/local/anaconda3/202309/bin/conda shell.bash hook)"
conda activate fit2

export GLOBAL_DIR="/cosma/home/dp322/dc-guan2/fitting/pipeline"
python -u $GLOBAL_DIR/src/inference.py -config $GLOBAL_DIR/$CONFIG_FILE
# Use the following if there's no intention to run it in your pc
# python -u $GLOBAL_DIR/src/inference.py -config $GLOBAL_DIR/$CONFIG_FILE -ncpus $SLURM_CPUS_PER_TASK
