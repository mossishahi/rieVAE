#!/bin/bash
# Graph convergence experiment: torus T^2.
# Same as sphere but on a K=0 manifold.
#SBATCH -o /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_conv_torus_%j.job
#SBATCH -e /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_conv_torus_err_%j.job
#SBATCH --job-name=rieVAE-conv-torus
#SBATCH --mem=32G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
set -euo pipefail

IMAGE_SQSH="/home/icb/mostafa.shahhosseini/global_updated.sqsh"
ENROOT_NAME="rieVAE_conv_torus_${SLURM_JOB_ID}"
RIEVAE_DIR="$HOME/code/repos/rieVAE"
OUTDIR="$RIEVAE_DIR/runs/convergence"
mkdir -p "$OUTDIR"

enroot create -n "$ENROOT_NAME" "$IMAGE_SQSH"
enroot start \
  --rw \
  --mount "$HOME:$HOME" \
  --env HOME="$HOME" \
  "$ENROOT_NAME" \
  bash -lc "
    source /opt/global/bin/activate
    pip install -e $RIEVAE_DIR --no-deps --quiet
    python -u $RIEVAE_DIR/scripts/train_convergence.py \
      --manifold torus \
      --n_points 3000 \
      --ambient_dim 50 \
      --noise 0.01 \
      --dim_latent 8 \
      --dim_edge 2 \
      --k_neighbors 8 \
      --n_iterations 50 \
      --n_mstep_epochs 500 \
      --lr 1e-3 \
      --beta_node_kl 1e-2 \
      --lambda_riem 10.0 \
      --beta_edge_kl 1e-3 \
      --convergence_threshold 0.01 \
      --seed 42 \
      --out_dir $OUTDIR \
      --device cuda
  "
enroot remove -f "$ENROOT_NAME" || true
