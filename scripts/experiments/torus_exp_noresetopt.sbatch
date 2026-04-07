#!/bin/bash
# Torus T^2 experiment: no optimizer reset (accumulated gradient info helps
# the edge encoder learn the torus log maps on a topologically non-trivial manifold).
# Distance clipping is kept (fixes the 2-cycle oscillation).
# Expected runtime: ~4-6h on GPU.
#SBATCH -o /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_torus_%j.job
#SBATCH -e /home/icb/mostafa.shahhosseini/code/repos/rieVAE/slurm_torus_err_%j.job
#SBATCH --job-name=rieVAE-torus-noreset
#SBATCH --mem=32G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
set -euo pipefail

IMAGE_SQSH="/home/icb/mostafa.shahhosseini/global_updated.sqsh"
ENROOT_NAME="rieVAE_torus_noreset_${SLURM_JOB_ID}"
RIEVAE_DIR="$HOME/code/repos/rieVAE"
OUTDIR="$RIEVAE_DIR/runs/torus_noreset"
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
      --wandb_run_name torus_d2_iter100_lriem0.1_clip5_noreset
  "
enroot remove -f "$ENROOT_NAME" || true
