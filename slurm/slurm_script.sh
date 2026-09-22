#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --job-name=eg
#SBATCH --clusters=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=devel
#SBATCH --gres=gpu:1
# --constraint=gpu_sku:H100

# Generic submission script: runs any `rmlt` workflow script on the compute node.
#
# Submit through the rmlt command (recommended - it checks the script name and finds
# this file for you):
#
#     rmlt slurm run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml
#
# or call sbatch directly:
#
#     sbatch slurm/slurm_script.sh run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml
#
# ===== EDIT THE #SBATCH LINES ABOVE TO SUIT THE JOB =====
#
# SLURM reads its directives from this file: partition / time / memory / GPU / cluster 
# can be changed by editing the header above. They can also be overridden per submission on the command line:
#
#     rmlt slurm --time=12:00:00 --partition=medium run-mlip-md inputs/md/x.yaml
#     rmlt slurm --gres=none run-reaction-barrier-comparison inputs/irc/x.yaml
#
# Example jobs (edit the header to match the required compute):
#
#     rmlt slurm run-reaction-barrier-comparison inputs/irc/CPD_MVK_DA_IRC_barrier.yaml
#     rmlt slurm run-test-set-comparison inputs/test/guajadial_DA_FT50_test.yaml
#     rmlt slurm run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml
#     rmlt slurm run-mlip-ts-opt inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch $0 <rmlt-script> <input.yaml> [args...]" >&2
    echo "   eg: sbatch $0 run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml" >&2
    echo "   (or submit through rmlt: rmlt slurm <rmlt-script> <input.yaml>)" >&2
    exit 2
fi

# load the conda env that has rmlt installed. Both steps are guarded so that the script
# is also runnable off-cluster (used by the tests, and handy for a local test).
if command -v module >/dev/null 2>&1; then
    module load Anaconda3
fi
if [ -n "${DATA:-}" ]; then
    source activate "$DATA/latest_mace"
fi

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# a #SBATCH directive cannot be set from an argument, so --job-name is whatever the
# header says; correct it here to make the run identifiable in squeue / sacct
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update jobid="$SLURM_JOB_ID" jobname="$1-$(basename "${2%.yaml}")" 2>/dev/null || true
fi

# run from the directory the job was submitted from: the input YAML path is resolved
# relative to the working directory (data / models / out are rooted at $DATA on the
# cluster, so they do not depend on this)
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

echo "Running on $(hostname) in $(pwd): rmlt $*"
rmlt "$@"
