#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM Job Configuration
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=pytorch260_benchmark
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# ─────────────────────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────────────────────
module reset
module list

export APPTAINER_CACHEDIR=$HOME/Pytorch_Install/cache
export TMPDIR=$HOME/Pytorch_Install/tmp

CUSTOM_SIF=$HOME/Pytorch_Install/pytorch260.sif
NGC_SIF=/sw/user/NGC_containers/builds/pytorch_25.08-py3-libfabric-2.3.1-build-ofi.sif
BENCHMARK=$HOME/Pytorch_Install/GPU_Benchmark.py

# ─────────────────────────────────────────────────────────────
# Sanity Checks
# ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Node         : $SLURMD_NODENAME"
echo "  Started at   : $(date)"
echo "  Arch         : $(uname -m)"
echo "============================================================"

echo ""
echo "Checking required files..."

if [ ! -f "$CUSTOM_SIF" ]; then
    echo "ERROR: Custom container not found at $CUSTOM_SIF"
    exit 1
fi

if [ ! -f "$BENCHMARK" ]; then
    echo "ERROR: Benchmark script not found at $BENCHMARK"
    exit 1
fi

echo "  ✓ Custom SIF found   : $CUSTOM_SIF"
echo "  ✓ Benchmark found    : $BENCHMARK"
echo "  ✓ NGC SIF target     : $NGC_SIF"

# ─────────────────────────────────────────────────────────────
# Run 1: Custom PyTorch Container (NGC base)
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RUN 1: Custom PyTorch Container (NGC 25.01 base)"
echo "============================================================"
echo "Started at: $(date)"

time apptainer exec --nv \
    --bind $HOME/Pytorch_Install:/workspace \
    $CUSTOM_SIF \
    /opt/venv/bin/python /workspace/GPU_Benchmark.py

echo "Finished at: $(date)"

echo ""
echo "Cooling down between runs (30 seconds)..."
sleep 30

# ─────────────────────────────────────────────────────────────
# Run 2: NVIDIA NGC Container (pre-installed on DeltaAI)
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RUN 2: NVIDIA NGC Container (pre-installed)"
echo "============================================================"
echo "Started at: $(date)"

time apptainer exec --nv \
    --bind $HOME/Pytorch_Install:/workspace \
    $NGC_SIF \
    python3 /workspace/GPU_Benchmark.py

echo "Finished at: $(date)"

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All runs complete."
echo "  Results saved to: benchmark_${SLURM_JOB_ID}.out"
echo "  Errors saved to : benchmark_${SLURM_JOB_ID}.err"
echo "  Finished at     : $(date)"
echo "============================================================"
