#!/bin/bash
# Clifford flat torus T^2 experiment (CORRECT version).
# Uses the Clifford embedding (R^4) which is genuinely flat (K=0 everywhere).
# The flat geodesic formula d=sqrt((R*dtheta)^2+(r*dphi)^2) is EXACT here.
# Previous experiments used the standard R^3 torus (nonzero varying curvature).
#
# Expected curvature proxy (reconstruction regime, Proposition 4):
#   kappa_hat* = sqrt(1/R^2 + 1/r^2) * sqrt(4/G) = sqrt(5/50) ~ 0.316
# Note: kappa_hat -> 0 is NOT achievable even for a flat torus because the
# Clifford embedding has nonzero second fundamental form h (extrinsic curvature).
# Values near 0.316 indicate a well-calibrated decoder.
#SBATCH -o /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_torus_clifford_%j.job
#SBATCH -e /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_torus_clifford_err_%j.job
#SBATCH --job-name=rieVAE-torus-clifford
#SBATCH --mem=32G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
set -euo pipefail

IMAGE_SQSH="/home/icb/mostafa.shahhosseini/global_updated.sqsh"
ENROOT_NAME="rieVAE_torus_clifford_${SLURM_JOB_ID}"
RIEVAE_DIR="$HOME/code/repos/rieVAE"
OUTDIR="$RIEVAE_DIR/runs/torus_clifford"
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
    python -u $RIEVAE_DIR/scripts/train_torus.py \
      --n_points 3000 \
      --ambient_dim 50 \
      --major_radius 2.0 \
      --minor_radius 1.0 \
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
      --distance_clip_factor 5.0 \
      --no_reset_optimizer \
      --seed 42 \
      --out_dir $OUTDIR \
      --device cuda \
      --use_wandb \
      --wandb_project rieVAE \
      --wandb_run_name torus_clifford_d2_iter100_noreset_clip5
  "
enroot remove -f "$ENROOT_NAME" || true
