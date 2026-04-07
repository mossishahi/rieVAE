#!/bin/bash
# Sphere S^2 experiment: SCR-VAE vs Vanilla VAE baseline.
# Key changes vs. previous run:
#   - dim_latent=2 (matches S^2 intrinsic dimension)
#   - 100 self-consistency iterations, no early stop
#   - 1000 M-step epochs per iteration
#   - lambda_riem=0.1 (smaller to not dominate reconstruction)
#   - SVD Stiefel retraction (more numerically stable than QR)
#   - Full diagnostics logged every 100 epochs + per-iteration to wandb
# Expected runtime: ~4-6h on GPU.
#SBATCH -o /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_sphere_%j.job
#SBATCH -e /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_sphere_err_%j.job
#SBATCH --job-name=rieVAE-sphere-v2
#SBATCH --mem=32G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
set -euo pipefail

IMAGE_SQSH="/home/icb/mostafa.shahhosseini/global_updated.sqsh"
ENROOT_NAME="rieVAE_sphere_v2_${SLURM_JOB_ID}"
RIEVAE_DIR="$HOME/code/repos/rieVAE"
OUTDIR="$RIEVAE_DIR/runs/sphere_v2"
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
    pip install wandb --quiet || true
    python -u $RIEVAE_DIR/scripts/train_sphere.py \
      --n_points 3000 \
      --ambient_dim 50 \
      --radius 1.0 \
      --noise 0.01 \
      --dim_latent 2 \
      --dim_edge 2 \
      --k_neighbors 10 \
      --n_iterations 100 \
      --n_mstep_epochs 1000 \
      --n_baseline_epochs 100000 \
      --lr 1e-3 \
      --beta_node_kl 1e-2 \
      --lambda_riem 0.1 \
      --beta_edge_kl 1e-3 \
      --lambda_decorr 0.0 \
      --n_eval_pairs 2000 \
      --max_triangles 1000 \
      --log_interval 100 \
      --seed 42 \
      --out_dir $OUTDIR \
      --device cuda \
      --distance_clip_factor 5.0 \
      --use_wandb \
      --wandb_project rieVAE \
      --wandb_run_name sphere_d2_iter100_lriem0.1_resetopt_clip5
  "
enroot remove -f "$ENROOT_NAME" || true
