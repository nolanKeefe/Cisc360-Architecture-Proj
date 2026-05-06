#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM Job Configuration
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=pytorch260_build
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:02:00
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err

# ─────────────────────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────────────────────
module reset

export APPTAINER_CACHEDIR=$HOME/Pytorch_Install/cache
export TMPDIR=$HOME/Pytorch_Install/tmp
export APPTAINER_DOCKER_ARCHITECTURE=arm64
mkdir -p $APPTAINER_CACHEDIR $TMPDIR

# ─────────────────────────────────────────────────────────────
# Info
# ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  Started at : $(date)"
echo "  Arch       : $(uname -m)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────
echo ""
echo "── Building pytorch260.sif ─────────────────────────────────"

apptainer build --fakeroot \
    $HOME/Pytorch_Install/pytorch260.sif \
    $HOME/Pytorch_Install/pytorch260.def

# ─────────────────────────────────────────────────────────────
# Confirm output
# ─────────────────────────────────────────────────────────────
if [ -f "$HOME/Pytorch_Install/pytorch260.sif" ]; then
    echo ""
    echo "  ✓ Build succeeded!"
    echo "  SIF size: $(du -sh $HOME/Pytorch_Install/pytorch260.sif | cut -f1)"
    echo ""
    echo "  Next step: sbatch $HOME/Pytorch_Install/test_container.sh"
else
    echo ""
    echo "  ✗ Build FAILED — check build_${SLURM_JOB_ID}.err for details"
fi

echo ""
echo "============================================================"
echo "  Finished at : $(date)"
echo "============================================================"
