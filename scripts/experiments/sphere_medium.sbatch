#!/bin/bash
# Sphere S^2 MEDIUM-SCALE experiment -- ~2-3h on GPU.
# Designed to satisfy the kappa_hat calibration condition:
#   baseline_kappa >= 3 * kappa_theory  =>  n_baseline_epochs >= 30 * n_points
#
# Parameters:
#   N = 2000  =>  30 * 2000 = 60000 minimum baseline epochs  (using 60000)
#   SCR-VAE: 25 iterations x 800 epochs = 20000 total epochs
#
# Expected outputs (qualitative -- exact numbers differ from full run):
#   Gram error:           0.000000 throughout
#   baseline kappa_hat:   > 1.5  (well above theory ~0.489)
#   SCR-VAE kappa_hat:    < baseline  (improvement direction correct)
#   SCR-VAE riem MAE:     < baseline  (isometry improves)
#   Graph convergence:    f_t -> 0 by iter ~15-20
#
# Compare qualitatively to: runs/sphere_v2/results.json
# NOTE: full-scale results (N=3000, 100 iters, 1000 ep/iter) remain
#       the authoritative source for Table 1.
#SBATCH -o /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_sphere_medium_%j.job
#SBATCH -e /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_sphere_medium_err_%j.job
#SBATCH --job-name=rieVAE-sphere-medium
#SBATCH --mem=24G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
set -euo pipefail

IMAGE_SQSH="/home/icb/mostafa.shahhosseini/global_updated.sqsh"
ENROOT_NAME="rieVAE_sphere_medium_${SLURM_JOB_ID}"
RIEVAE_DIR="$HOME/code/repos/rieVAE"
OUTDIR="$RIEVAE_DIR/runs/sphere_medium"
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
    python -u $RIEVAE_DIR/scripts/train_sphere.py \
      --n_points 2000 \
      --ambient_dim 50 \
      --radius 1.0 \
      --noise 0.01 \
      --dim_latent 2 \
      --dim_edge 2 \
      --k_neighbors 10 \
      --n_iterations 25 \
      --n_mstep_epochs 800 \
      --n_baseline_epochs 60000 \
      --lr 1e-3 \
      --beta_node_kl 1e-2 \
      --lambda_riem 0.1 \
      --beta_edge_kl 1e-3 \
      --lambda_decorr 0.0 \
      --n_eval_pairs 1000 \
      --max_triangles 800 \
      --log_interval 100 \
      --seed 42 \
      --out_dir $OUTDIR \
      --device cuda \
      --distance_clip_factor 5.0
  "
enroot remove -f "$ENROOT_NAME" || true
