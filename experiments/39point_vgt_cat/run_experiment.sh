#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
: "${ANIMAL_DATA_ROOT:?Set ANIMAL_DATA_ROOT to the animal-motion checkout containing data/ and outputs/}"
: "${QIANJI_ROOT:?Set QIANJI_ROOT to the read-only QianJi checkout}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new versioned case directory}"
FINAL_OUTPUT_ROOT="$OUTPUT_ROOT"

VIDEO="${ANIMAL_DATA_ROOT}/data/processed/cat_walk_30fps_720p.mp4"
PREDICTIONS="${ANIMAL_DATA_ROOT}/outputs/zero_shot/cat_walk/cat_walk_30fps_720p_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5"
MESH="${ANIMAL_DATA_ROOT}/data/processed/小猫_assimp.glb"
CORRECTED="${ANIMAL_DATA_ROOT}/outputs/manual_correction/cat_walk/review_v2_migrated/keypoint_trajectory_2d_corrected.json"

PYTHON="${REPO_ROOT}/.venv/bin/python"
GENERATOR="${QIANJI_ROOT}/morph_generator/generate.py"
REACHABILITY="${QIANJI_ROOT}/controller/check_keypoint_reachability.py"
MORPHOLOGY="${QIANJI_ROOT}/controller/optimize_morphology_for_motion.py"
QIANJI_REACHABILITY_CORE="${QIANJI_ROOT}/controller/reachability.py"
QIANJI_SLIDE_ADAPTER="${QIANJI_ROOT}/controller/slide_control_adapter.py"
QIANJI_XML_CONVERTER="${QIANJI_ROOT}/mujoco_builder/json2xml_v7_perrod.py"
QIANJI_SCENE_BUILDER="${QIANJI_ROOT}/mujoco_builder/add_scene_to_xml.py"
FROZEN_ROBOT="${REPO_ROOT}/experiments/39point_vgt_cat/fixtures/cat_hunyuan_qianji_robot_12x30.json"
FROZEN_ROBOT_MANIFEST="${REPO_ROOT}/experiments/39point_vgt_cat/fixtures/cat_hunyuan_qianji_robot_12x30.manifest.json"

for command in git mamba jq ffmpeg ffprobe; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$command" >&2
    exit 2
  fi
done

for source in \
  "$VIDEO" "$PREDICTIONS" "$MESH" "$CORRECTED" \
  "$PYTHON" \
  "$FROZEN_ROBOT" "$FROZEN_ROBOT_MANIFEST" \
  "$GENERATOR" "$REACHABILITY" "$MORPHOLOGY" \
  "$QIANJI_REACHABILITY_CORE" "$QIANJI_SLIDE_ADAPTER" \
  "$QIANJI_XML_CONVERTER" "$QIANJI_SCENE_BUILDER"; do
  if [[ ! -f "$source" ]]; then
    printf 'Missing required source or command: %s\n' "$source" >&2
    exit 2
  fi
done
if [[ -e "$FINAL_OUTPUT_ROOT" ]]; then
  printf 'OUTPUT_ROOT already exists; choose a new versioned directory: %s\n' "$FINAL_OUTPUT_ROOT" >&2
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
mkdir -p \
  "$OUTPUT_ROOT/observation" \
  "$OUTPUT_ROOT/initial_model" \
  "$OUTPUT_ROOT/scales" \
  "$OUTPUT_ROOT/candidates" \
  "$OUTPUT_ROOT/landmarks" \
  "$OUTPUT_ROOT/control" \
  "$OUTPUT_ROOT/motion" \
  "$OUTPUT_ROOT/previews" \
  "$OUTPUT_ROOT/reports"

"$PYTHON" -m qianji_animal_motion.keypoints_39_cli \
  --video "$VIDEO" \
  --predictions "$PREDICTIONS" \
  --mesh "$MESH" \
  --corrected-spine "$CORRECTED" \
  --output "$OUTPUT_ROOT" \
  --reference-frame 152 \
  --confidence-threshold 0.30 \
  --mesh-generation-method hunyuan3d_from_video_frame

"$PYTHON" -m qianji_animal_motion.frozen_initial_robot \
  --mesh "$MESH" \
  --template "$FROZEN_ROBOT" \
  --template-manifest "$FROZEN_ROBOT_MANIFEST" \
  --output-dir "$OUTPUT_ROOT/initial_model"
cp "$OUTPUT_ROOT/initial_model/robot_base.json" \
  "$OUTPUT_ROOT/initial_model/robot.json"
(
  cd "$QIANJI_ROOT"
  mamba run -n biomimic python "$QIANJI_XML_CONVERTER" \
    -i "$OUTPUT_ROOT/initial_model/robot.json" \
    -o "$OUTPUT_ROOT/initial_model/robot.xml" \
    --model-name animal_motion_cat_39point_base
  mamba run -n biomimic python "$QIANJI_SCENE_BUILDER" \
    --input "$OUTPUT_ROOT/initial_model/robot.xml" \
    --output "$OUTPUT_ROOT/initial_model/robot_scene.xml"
  mamba run -n biomimic python "$REACHABILITY" \
    --robot-json "$OUTPUT_ROOT/initial_model/robot.json" \
    --auto-rig quadruped_bbox \
    --output-dir "$OUTPUT_ROOT/initial_model/bbox_seed"
)
cp "$OUTPUT_ROOT/initial_model/bbox_seed/rig_keypoints.json" \
  "$OUTPUT_ROOT/initial_model/rig_bbox.json"
cp "$OUTPUT_ROOT/initial_model/robot.xml" \
  "$OUTPUT_ROOT/initial_model/robot_base.xml"
cp "$OUTPUT_ROOT/initial_model/robot_scene.xml" \
  "$OUTPUT_ROOT/initial_model/robot_base_scene.xml"

for scale_code in 005 010 015; do
  case "$scale_code" in
    005) motion_scale="0.05" ;;
    010) motion_scale="0.10" ;;
    015) motion_scale="0.15" ;;
  esac
  scale_root="$OUTPUT_ROOT/scales/scale_${scale_code}"
  "$PYTHON" -m qianji_animal_motion.lift_39_cli \
    --trajectory-39 "$OUTPUT_ROOT/observation/keypoint_trajectory_2d_39.json" \
    --corrected-spine "$CORRECTED" \
    --robot-json "$OUTPUT_ROOT/initial_model/robot_base.json" \
    --rig "$OUTPUT_ROOT/initial_model/rig_bbox.json" \
    --output "$scale_root/landmarks" \
    --reference-frame 152 \
    --motion-scale "$motion_scale" \
    --case-root "$OUTPUT_ROOT"
  "$PYTHON" -m qianji_animal_motion.vgt_control_cli \
    --robot-json "$OUTPUT_ROOT/initial_model/robot.json" \
    --bbox-rig "$OUTPUT_ROOT/initial_model/rig_bbox.json" \
    --neutral-landmarks "$scale_root/landmarks/neutral_landmarks_39.json" \
    --motion-39 "$scale_root/landmarks/keypoint_motion_3d_39.json" \
    --output "$scale_root/control" \
    --contraction-fractions 0.0 0.1
done

run_reachability_candidate() {
  local candidate_id="$1"
  local scale_code="$2"
  local motion_scale="$3"
  local contraction="$4"
  local rig_variant="$5"
  local allow_contraction="$6"
  local candidate_root="$OUTPUT_ROOT/candidates/$candidate_id"
  local scale_root="$OUTPUT_ROOT/scales/scale_${scale_code}"
  local robot_suffix="000"
  local rig_path="$scale_root/control/rig_bbox.json"
  local target_path="$scale_root/control/target_control_motion_bbox.json"
  if [[ "$contraction" == "0.1" ]]; then
    robot_suffix="010"
  fi
  if [[ "$rig_variant" == "motion_informed" ]]; then
    rig_path="$scale_root/control/rig_motion_informed.json"
    target_path="$scale_root/control/target_control_motion_motion_informed.json"
  fi
  local robot_path="$scale_root/control/robot_contraction_${robot_suffix}.json"
  local reachability_args=(
    --robot-json "$robot_path"
    --rig-config "$rig_path"
    --keypoint-motion "$target_path"
    --output-dir "$candidate_root/reachability"
  )
  if [[ "$allow_contraction" == "yes" ]]; then
    reachability_args+=(--allow-contraction)
  fi
  (
    cd "$QIANJI_ROOT"
    mamba run -n biomimic python "$REACHABILITY" "${reachability_args[@]}"
  )
  jq -n \
    --arg candidate_id "$candidate_id" \
    --argjson motion_scale "$motion_scale" \
    --argjson contraction_fraction "$contraction" \
    --arg rig_variant "$rig_variant" \
    --arg landmark_dir "$scale_root/landmarks" \
    --arg control_map "$scale_root/control/control_map_${rig_variant}.json" \
    --arg run_summary "$candidate_root/reachability/run_summary.json" \
    --arg reachability_report "$candidate_root/reachability/reachability_report.json" \
    --arg robot "$robot_path" \
    --arg rig "$rig_path" \
    --arg desired_motion "$target_path" \
    --arg projected_motion "$candidate_root/reachability/projected_keypoint_motion.json" \
    --arg site_npz "$candidate_root/reachability/feasible_site_targets.npz" \
    '{
      candidate_id: $candidate_id,
      expected_frames: 272,
      parameters: {
        motion_scale: $motion_scale,
        contraction_fraction: $contraction_fraction,
        rig_variant: $rig_variant,
        morphology_variant: "base"
      },
      publication: {
        landmark_dir: $landmark_dir,
        control_map: $control_map
      },
      artifacts: {
        run_summary: $run_summary,
        reachability_report: $reachability_report,
        robot: $robot,
        rig: $rig,
        desired_motion: $desired_motion,
        projected_motion: $projected_motion,
        site_npz: $site_npz
      }
    }' > "$candidate_root/candidate.json"
}

run_reachability_candidate scale_005_bbox_base_c000 005 0.05 0.0 bbox no
run_reachability_candidate scale_010_bbox_base_c000 010 0.10 0.0 bbox no
run_reachability_candidate scale_015_bbox_base_c000 015 0.15 0.0 bbox no
run_reachability_candidate scale_010_bbox_base_c010 010 0.10 0.1 bbox yes
run_reachability_candidate scale_010_motion_informed_base_c000 010 0.10 0.0 motion_informed no
run_reachability_candidate scale_010_motion_informed_base_c010 010 0.10 0.1 motion_informed yes

morph_candidate="scale_010_motion_informed_morph_c000"
morph_root="$OUTPUT_ROOT/candidates/$morph_candidate"
scale_root="$OUTPUT_ROOT/scales/scale_010"
morph_robot="$scale_root/control/robot_contraction_000.json"
morph_rig="$scale_root/control/rig_motion_informed.json"
morph_target="$scale_root/control/target_control_motion_motion_informed.json"
(
  cd "$QIANJI_ROOT"
  mamba run -n biomimic python "$MORPHOLOGY" \
    --robot-json "$morph_robot" \
    --rig-config "$morph_rig" \
    --keypoint-motion "$morph_target" \
    --base-reachability-report \
      "$OUTPUT_ROOT/candidates/scale_010_motion_informed_base_c000/reachability/reachability_report.json" \
    --output-dir "$morph_root/morphology" \
    --name animal_motion_cat_39point_optimized
)
jq '{summary: .summary}' \
  "$morph_root/morphology/reachability_after/reachability_report.json" \
  > "$morph_root/run_summary.json"
jq -n \
  --arg candidate_id "$morph_candidate" \
  --arg landmark_dir "$scale_root/landmarks" \
  --arg control_map "$scale_root/control/control_map_motion_informed.json" \
  --arg run_summary "$morph_root/run_summary.json" \
  --arg reachability_report "$morph_root/morphology/reachability_after/reachability_report.json" \
  --arg robot "$morph_root/morphology/robot_optimized.json" \
  --arg rig "$morph_rig" \
  --arg desired_motion "$morph_target" \
  --arg projected_motion "$morph_root/morphology/reachability_after/projected_keypoint_motion.json" \
  --arg site_npz "$morph_root/morphology/reachability_after/feasible_site_targets.npz" \
  --arg morphology_report "$morph_root/morphology/morph_optimization_report.json" \
  '{
    candidate_id: $candidate_id,
    expected_frames: 272,
    parameters: {
      motion_scale: 0.10,
      contraction_fraction: 0.0,
      rig_variant: "motion_informed",
      morphology_variant: "optimized"
    },
    publication: {
      landmark_dir: $landmark_dir,
      control_map: $control_map
    },
    artifacts: {
      run_summary: $run_summary,
      reachability_report: $reachability_report,
      robot: $robot,
      rig: $rig,
      desired_motion: $desired_motion,
      projected_motion: $projected_motion,
      site_npz: $site_npz,
      morphology_report: $morphology_report
    }
  }' > "$morph_root/candidate.json"

"$PYTHON" "$REPO_ROOT/experiments/39point_vgt_cat/compare_candidates.py" \
  "$OUTPUT_ROOT"/candidates/* \
  --output "$OUTPUT_ROOT/reports/reachability_report.json"

jq '.selected_candidate' "$OUTPUT_ROOT/reports/reachability_report.json" \
  > "$OUTPUT_ROOT/reports/selected_candidate.json"
selected_root="$(jq -r '.selected_candidate.candidate_root' "$OUTPUT_ROOT/reports/reachability_report.json")"
selected_root="$OUTPUT_ROOT/$selected_root"
selected_meta="$selected_root/candidate.json"
landmark_dir="$(jq -r '.publication.landmark_dir' "$selected_meta")"
control_map="$(jq -r '.publication.control_map' "$selected_meta")"
selected_robot="$(jq -r '.artifacts.robot' "$selected_meta")"
selected_rig="$(jq -r '.artifacts.rig' "$selected_meta")"
selected_desired="$(jq -r '.artifacts.desired_motion' "$selected_meta")"
selected_projected="$(jq -r '.artifacts.projected_motion' "$selected_meta")"
selected_npz="$(jq -r '.artifacts.site_npz' "$selected_meta")"

cp "$landmark_dir/neutral_landmarks_39.json" "$OUTPUT_ROOT/landmarks/"
cp "$landmark_dir/keypoint_motion_3d_39.json" "$OUTPUT_ROOT/landmarks/"
cp "$landmark_dir/lift_39_report.json" "$OUTPUT_ROOT/landmarks/"
cp "$control_map" "$OUTPUT_ROOT/control/vgt_control_map.json"
cp "$selected_desired" "$OUTPUT_ROOT/control/target_control_keypoint_motion.json"
cp "$selected_projected" "$OUTPUT_ROOT/control/projected_control_keypoint_motion.json"
cp "$selected_rig" "$OUTPUT_ROOT/initial_model/rig_keypoints.json"
cp "$selected_robot" "$OUTPUT_ROOT/initial_model/robot.json"

(
  cd "$QIANJI_ROOT"
  mamba run -n biomimic python "$QIANJI_XML_CONVERTER" \
    -i "$OUTPUT_ROOT/initial_model/robot.json" \
    -o "$OUTPUT_ROOT/initial_model/robot.xml" \
    --model-name animal_motion_cat_39point_selected
  mamba run -n biomimic python "$QIANJI_SCENE_BUILDER" \
    --input "$OUTPUT_ROOT/initial_model/robot.xml" \
    --output "$OUTPUT_ROOT/initial_model/robot_scene.xml"
)
"$PYTHON" \
  "$REPO_ROOT/experiments/39point_vgt_cat/record_robot_conversion.py" \
  "$OUTPUT_ROOT"

"$PYTHON" -m qianji_animal_motion.vgt_sequence_cli \
  --site-npz "$selected_npz" \
  --robot-json "$OUTPUT_ROOT/initial_model/robot.json" \
  --rig "$OUTPUT_ROOT/initial_model/rig_keypoints.json" \
  --desired-control-motion "$OUTPUT_ROOT/control/target_control_keypoint_motion.json" \
  --projected-control-motion "$OUTPUT_ROOT/control/projected_control_keypoint_motion.json" \
  --selected-candidate "$OUTPUT_ROOT/reports/selected_candidate.json" \
  --output "$OUTPUT_ROOT/motion" \
  --expected-frames 272 \
  --expected-sites 12 \
  --expected-rods 30 \
  --width 1246 \
  --height 720 \
  --fps 30
cp "$OUTPUT_ROOT/motion/vgt_three_view.png" "$OUTPUT_ROOT/previews/"
cp "$OUTPUT_ROOT/motion/vgt_isometric.png" "$OUTPUT_ROOT/previews/"
cp "$OUTPUT_ROOT/motion/vgt_motion_30fps.mp4" "$OUTPUT_ROOT/previews/"

mamba run -n biomimic python \
  "${REPO_ROOT}/src/qianji_animal_motion/experiment_provenance.py" \
  --repository-root "$REPO_ROOT" \
  --qianji-root "$QIANJI_ROOT" \
  --script "${REPO_ROOT}/experiments/39point_vgt_cat/run_experiment.sh" \
  --script "${REPO_ROOT}/experiments/39point_vgt_cat/compare_candidates.py" \
  --script "${REPO_ROOT}/experiments/39point_vgt_cat/verify_case.py" \
  --script "${REPO_ROOT}/experiments/39point_vgt_cat/record_robot_conversion.py" \
  --script "${REPO_ROOT}/src/qianji_animal_motion/frozen_initial_robot.py" \
  --script "$GENERATOR" \
  --script "$REACHABILITY" \
  --script "$MORPHOLOGY" \
  --script "$QIANJI_REACHABILITY_CORE" \
  --script "$QIANJI_SLIDE_ADAPTER" \
  --script "$QIANJI_XML_CONVERTER" \
  --script "$QIANJI_SCENE_BUILDER" \
  --output "$OUTPUT_ROOT/reports/experiment_provenance.json"

"$PYTHON" "$REPO_ROOT/experiments/39point_vgt_cat/verify_case.py" "$OUTPUT_ROOT"
if [[ -e "$FINAL_OUTPUT_ROOT" ]]; then
  printf 'OUTPUT_ROOT appeared during the run: %s\n' "$FINAL_OUTPUT_ROOT" >&2
  exit 2
fi
mv "$CASE_STAGING_ROOT" "$FINAL_OUTPUT_ROOT"
CASE_STAGING_ROOT=""
trap - EXIT HUP INT TERM
printf 'Complete 39-point VGT case: %s\n' "$FINAL_OUTPUT_ROOT"
