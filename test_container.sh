#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM Job Configuration
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=pytorch260_test
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --time=00:10:00
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

# ─────────────────────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────────────────────
module reset

CUSTOM_SIF=$HOME/Pytorch_Install/pytorch260.sif

# ─────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  Started at : $(date)"
echo "  Arch       : $(uname -m)"
echo "============================================================"

if [ ! -f "$CUSTOM_SIF" ]; then
    echo "ERROR: SIF not found at $CUSTOM_SIF"
    exit 1
fi

echo "  ✓ SIF found: $CUSTOM_SIF"

# ─────────────────────────────────────────────────────────────
# Test 1: Python and PyTorch import
# ─────────────────────────────────────────────────────────────
echo ""
echo "── Test 1: PyTorch version and CUDA availability ───────────"

apptainer exec --nv \
    $CUSTOM_SIF \
    /opt/venv/bin/python -c "
import torch
print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())
print('CUDA version    :', torch.version.cuda)
print('GPU name        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — check --nv flag')
print('GPU count       :', torch.cuda.device_count())
"

# ─────────────────────────────────────────────────────────────
# Test 2: Small GPU computation
# ─────────────────────────────────────────────────────────────
echo ""
echo "── Test 2: Small GPU computation ───────────────────────────"

apptainer exec --nv \
    $CUSTOM_SIF \
    /opt/venv/bin/python -c "
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on:', device)

A = torch.randn(1000, 1000, device=device)
B = torch.randn(1000, 1000, device=device)
C = torch.matmul(A, B)

print('Result shape    :', C.shape)
print('Result device   :', C.device)
print('Result dtype    :', C.dtype)
print('Computation     : PASSED ✓')
"

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Test complete."
echo "  Check test_${SLURM_JOB_ID}.out for results."
echo "  If both tests passed, run: sbatch benchmark.sh"
echo "  Finished at : $(date)"
echo "============================================================"
