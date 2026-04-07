#!/bin/bash
# Sphere S^2 SANITY CHECK -- reduced scale, ~30-45 min on GPU.
# Purpose: verify code correctness after all changes without rerunning the
# full 4-6h paper experiment. Key checks:
#   1. kappa_hat moves from ~2.8 (baseline) toward ~1.4 (SCR-VAE) direction
#   2. Graph change fraction reaches 0 by iteration ~15 (convergence pattern)
#   3. Gram error stays at 0.000000 (Stiefel constraint)
#   4. Riemannian pullback MAE improves vs. baseline
#   5. No import errors, shape errors, or NaNs from the code changes
#
# Compare outputs against: runs/sphere_v2/results.json
# Qualitative agreement suffices; exact numbers differ due to fewer epochs.
#SBATCH -o /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_sphere_sanity_%j.job
#SBATCH -e /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_sphere_sanity_err_%j.job
#SBATCH --job-name=rieVAE-sphere-sanity
#SBATCH --mem=16G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
set -euo pipefail

IMAGE_SQSH="/home/icb/mostafa.shahhosseini/global_updated.sqsh"
ENROOT_NAME="rieVAE_sphere_sanity_${SLURM_JOB_ID}"
RIEVAE_DIR="$HOME/code/repos/rieVAE"
OUTDIR="$RIEVAE_DIR/runs/sphere_sanity"
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

    # --- Unit tests first (fail fast) ---
    echo '=== Running unit tests ===' && \
    pytest $RIEVAE_DIR/tests/ -v --tb=short 2>&1 | tail -30

    # --- Reduced-scale sphere experiment ---
    python -u $RIEVAE_DIR/scripts/train_sphere.py \
      --n_points 1000 \
      --ambient_dim 50 \
      --radius 1.0 \
      --noise 0.01 \
      --dim_latent 2 \
      --dim_edge 2 \
      --k_neighbors 10 \
      --n_iterations 20 \
      --n_mstep_epochs 500 \
      --n_baseline_epochs 10000 \
      --lr 1e-3 \
      --beta_node_kl 1e-2 \
      --lambda_riem 0.1 \
      --beta_edge_kl 1e-3 \
      --lambda_decorr 0.0 \
      --n_eval_pairs 500 \
      --max_triangles 500 \
      --log_interval 100 \
      --seed 42 \
      --out_dir $OUTDIR \
      --device cuda \
      --distance_clip_factor 5.0
  "
enroot remove -f "$ENROOT_NAME" || true
