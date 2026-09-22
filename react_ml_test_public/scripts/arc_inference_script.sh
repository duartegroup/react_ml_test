#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --job-name=arc_inference_script
#SBATCH --clusters=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=devel
#SBATCH --gres=gpu:1
# --constraint='gpu_sku:H100' --constraint='gpu_mem:80GB' 
# --constraint=gpu_sku:H100
# --constraint='gpu_sku:H100' --constraint='nvlink:2.0' --constraint='gpu_mem:80GB'

# nvidia-smi
module load Anaconda3
# source activate $DATA/mlptrain-mace
source activate $DATA/latest_mace

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ===== CHANGE THE SCRIPT TO RUN HERE =====

# reaction barriers
rmlt run-reaction-barrier-comparison inputs/irc/CPD_MVK_DA_IRC_barrier.yaml

# test set comparison
# rmlt run-test-set-comparison inputs/test/guajadial_DA_FT50_test.yaml

# md
# rmlt run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml

# ts opt
# rmlt run-mlip-ts-opt inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml
