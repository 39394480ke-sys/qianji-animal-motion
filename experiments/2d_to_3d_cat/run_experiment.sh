#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ANIMAL_DATA_ROOT:?Set ANIMAL_DATA_ROOT to the main animal-motion checkout}"
: "${QIANJI_ROOT:?Set QIANJI_ROOT to the QianJi checkout}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/experiments/2d_to_3d_cat}"

MESH="${ANIMAL_DATA_ROOT}/data/processed/小猫_assimp.glb"
TRAJECTORY="${ANIMAL_DATA_ROOT}/outputs/manual_correction/cat_walk/review_v2_migrated/keypoint_trajectory_2d_corrected.json"
LIFT_CLI="${REPO_ROOT}/.venv/bin/qianji-lift-keypoints"
ROBOT="${OUTPUT_ROOT}/morphology/robot.json"
RIG="${OUTPUT_ROOT}/rig_seed/rig_keypoints.json"

for source in "$MESH" "$TRAJECTORY" "$LIFT_CLI"; do
  if [[ ! -f "$source" ]]; then
    printf 'Missing required source: %s\n' "$source" >&2
    exit 2
  fi
done
if [[ ! -f "${QIANJI_ROOT}/morph_generator/generate.py" ]]; then
  printf 'QIANJI_ROOT is not a QianJi checkout: %s\n' "$QIANJI_ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'OUTPUT_ROOT already exists; choose a new directory: %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$QIANJI_ROOT"

mamba run -n biomimic python morph_generator/generate.py 3d-mesh \
  --mesh "$MESH" \
  --preset abstract \
  --name animal_motion_cat \
  --output-dir "${OUTPUT_ROOT}/morphology"

mamba run -n biomimic python controller/check_keypoint_reachability.py \
  --robot-json "$ROBOT" \
  --auto-rig quadruped_bbox \
  --output-dir "${OUTPUT_ROOT}/rig_seed"

for scale in 0.10 0.25 0.50; do
  SCALE_ROOT="${OUTPUT_ROOT}/scale_${scale}"
  "$LIFT_CLI" \
    --trajectory "$TRAJECTORY" \
    --robot-json "$ROBOT" \
    --rig "$RIG" \
    --output "$SCALE_ROOT" \
    --motion-scale "$scale"

  mamba run -n biomimic python controller/check_keypoint_reachability.py \
    --robot-json "$ROBOT" \
    --rig-config "$RIG" \
    --keypoint-motion "${SCALE_ROOT}/keypoint_motion.json" \
    --output-dir "${SCALE_ROOT}/reachability"

  mamba run -n biomimic python controller/visualize_keypoint_motion.py \
    --keypoint-motion "${SCALE_ROOT}/keypoint_motion.json" \
    --robot-json "$ROBOT" \
    --rig-config "$RIG" \
    --output-dir "${SCALE_ROOT}/preview" \
    --record-video \
    --hide-frame-labels \
    --hide-role-labels
done

cd "$REPO_ROOT"
"${REPO_ROOT}/.venv/bin/python" \
  experiments/2d_to_3d_cat/summarize_results.py \
  --output-root "$OUTPUT_ROOT" \
  --mesh "$MESH" \
  --trajectory "$TRAJECTORY"
