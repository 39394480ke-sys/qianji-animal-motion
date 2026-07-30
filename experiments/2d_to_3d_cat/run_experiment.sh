#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ANIMAL_DATA_ROOT:?Set ANIMAL_DATA_ROOT to the main animal-motion checkout}"
: "${QIANJI_ROOT:?Set QIANJI_ROOT to the QianJi checkout}"
FINAL_OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/experiments/2d_to_3d_cat}"

MESH="${ANIMAL_DATA_ROOT}/data/processed/小猫_assimp.glb"
TRAJECTORY="${ANIMAL_DATA_ROOT}/outputs/manual_correction/cat_walk/review_v2_migrated/keypoint_trajectory_2d_corrected.json"
PYTHON="${REPO_ROOT}/.venv/bin/python"

for command in git mamba ffmpeg ffprobe; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$command" >&2
    exit 2
  fi
done

for source in "$MESH" "$TRAJECTORY" "$PYTHON"; do
  if [[ ! -f "$source" ]]; then
    printf 'Missing required source: %s\n' "$source" >&2
    exit 2
  fi
done
QIANJI_SCRIPTS=(
  "${QIANJI_ROOT}/morph_generator/generate.py"
  "${QIANJI_ROOT}/controller/check_keypoint_reachability.py"
  "${QIANJI_ROOT}/controller/visualize_keypoint_motion.py"
)
for source in "${QIANJI_SCRIPTS[@]}"; do
  if [[ ! -f "$source" ]]; then
    printf 'Missing required QianJi script: %s\n' "$source" >&2
    exit 2
  fi
done
if [[ -e "$FINAL_OUTPUT_ROOT" ]]; then
  printf 'OUTPUT_ROOT already exists; choose a new directory: %s\n' "$FINAL_OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_PARENT="$(dirname "$FINAL_OUTPUT_ROOT")"
OUTPUT_NAME="$(basename "$FINAL_OUTPUT_ROOT")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT="$(cd "$OUTPUT_PARENT" && pwd -P)"
FINAL_OUTPUT_ROOT="${OUTPUT_PARENT}/${OUTPUT_NAME}"
CASE_STAGING_ROOT="$(mktemp -d "${OUTPUT_PARENT}/.${OUTPUT_NAME}.case-staging.XXXXXX")"

cleanup_staging() {
  if [[ -z "${CASE_STAGING_ROOT:-}" || ! -d "$CASE_STAGING_ROOT" ]]; then
    return
  fi
  case "$CASE_STAGING_ROOT" in
    "${OUTPUT_PARENT}/.${OUTPUT_NAME}.case-staging."*)
      rm -rf -- "$CASE_STAGING_ROOT"
      ;;
    *)
      printf 'Refusing to clean unexpected staging path: %s\n' "$CASE_STAGING_ROOT" >&2
      ;;
  esac
}
trap cleanup_staging EXIT
trap 'exit 130' HUP INT TERM

OUTPUT_ROOT="$CASE_STAGING_ROOT"
ROBOT="${OUTPUT_ROOT}/morphology/robot.json"
RIG="${OUTPUT_ROOT}/rig_seed/rig_keypoints.json"

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
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON" -m qianji_animal_motion.lift_cli \
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
    --video-fps 30 \
    --hide-frame-labels \
    --hide-role-labels
done

cd "$REPO_ROOT"
"$PYTHON" \
  experiments/2d_to_3d_cat/summarize_results.py \
  --output-root "$OUTPUT_ROOT" \
  --mesh "$MESH" \
  --trajectory "$TRAJECTORY"

mamba run -n biomimic python \
  "${REPO_ROOT}/src/qianji_animal_motion/experiment_provenance.py" \
  --repository-root "$REPO_ROOT" \
  --qianji-root "$QIANJI_ROOT" \
  --script "${REPO_ROOT}/experiments/2d_to_3d_cat/run_experiment.sh" \
  --script "${REPO_ROOT}/experiments/2d_to_3d_cat/summarize_results.py" \
  --script "${QIANJI_SCRIPTS[0]}" \
  --script "${QIANJI_SCRIPTS[1]}" \
  --script "${QIANJI_SCRIPTS[2]}" \
  --output "${OUTPUT_ROOT}/experiment_provenance.json"

if [[ -e "$FINAL_OUTPUT_ROOT" ]]; then
  printf 'OUTPUT_ROOT appeared during the run: %s\n' "$FINAL_OUTPUT_ROOT" >&2
  exit 2
fi
mv "$CASE_STAGING_ROOT" "$FINAL_OUTPUT_ROOT"
CASE_STAGING_ROOT=""
trap - EXIT HUP INT TERM
printf 'Published complete case: %s\n' "$FINAL_OUTPUT_ROOT"
