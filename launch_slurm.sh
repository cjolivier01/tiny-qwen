#!/bin/bash
# Launch Qwen3-VL training on Slurm via srun (one task per GPU).
#
# Slurm launches one process per GPU across potentially multiple nodes.
# Each process discovers its rank from SLURM_PROCID, local GPU from
# SLURM_LOCALID, and the master address from the first node in the
# allocation.  No torchrun required.
#
# Usage:
#   # 3 GPUs on 1 node:
#   ./launch_slurm.sh 3
#
#   # 8 GPUs across 2 nodes (4 per node):
#   ./launch_slurm.sh 8 --nodes 2
#
#   # With extra training args:
#   ./launch_slurm.sh 4 --steps 1000 --te-fp8 --tensorboard
#
#   # Benchmark mode:
#   ./launch_slurm.sh 3 --mode benchmark
#
#   # Dry run (print sbatch script without submitting):
#   DRY_RUN=1 ./launch_slurm.sh 3

set -euo pipefail

NUM_GPUS="${1:?Usage: $0 <num_gpus> [--nodes N] [--mode train|benchmark] [extra_args...]}"
shift

# Defaults
MODE="train"
NUM_NODES=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GPUS_PER_NODE=$(( NUM_GPUS / NUM_NODES ))
if (( GPUS_PER_NODE * NUM_NODES != NUM_GPUS )); then
    echo "ERROR: $NUM_GPUS GPUs is not evenly divisible by $NUM_NODES nodes" >&2
    exit 1
fi

PARTITION="${SLURM_PARTITION:-gnode-hi-pri}"
JOB_NAME="qwen3vl-${MODE}-${NUM_GPUS}gpu"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${WORK_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

if [[ "$MODE" == "benchmark" ]]; then
    SCRIPT="benchmark.py"
else
    SCRIPT="train_ddp.py"
fi

SBATCH_SCRIPT=$(cat <<'OUTER_EOF'
#!/bin/bash
#SBATCH --job-name=@@JOB_NAME@@
#SBATCH --partition=@@PARTITION@@
#SBATCH --nodes=@@NUM_NODES@@
#SBATCH --ntasks-per-node=@@GPUS_PER_NODE@@
#SBATCH --gres=gpu:gb300:@@GPUS_PER_NODE@@
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=@@LOG_DIR@@/%j_%x.out
#SBATCH --error=@@LOG_DIR@@/%j_%x.err
#SBATCH --exclusive

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
echo "Nodes: ${SLURM_JOB_NUM_NODES}, Tasks: ${SLURM_NTASKS}, GPUs/node: @@GPUS_PER_NODE@@"
echo "Nodelist: ${SLURM_NODELIST}"
echo "Script: @@SCRIPT@@"
echo "Extra args: @@EXTRA_ARGS@@"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "---"

# Master address = first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

cd "@@WORK_DIR@@"

# srun launches one task per GPU; the training script reads
# SLURM_PROCID (global rank), SLURM_LOCALID (local rank / GPU index),
# and SLURM_NTASKS (world size) to set up DDP.
srun python3 @@SCRIPT@@ @@EXTRA_ARGS@@

echo "=== Job complete ==="
OUTER_EOF
)

# Template substitution
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@JOB_NAME@@/${JOB_NAME}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@PARTITION@@/${PARTITION}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@NUM_NODES@@/${NUM_NODES}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@GPUS_PER_NODE@@/${GPUS_PER_NODE}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@LOG_DIR@@/${LOG_DIR}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@WORK_DIR@@/${WORK_DIR}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@SCRIPT@@/${SCRIPT}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT//@@EXTRA_ARGS@@/${EXTRA_ARGS[*]:-}}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "--- DRY RUN (sbatch script) ---"
    echo "$SBATCH_SCRIPT"
    echo "--- end ---"
else
    echo "$SBATCH_SCRIPT" | sbatch
fi
