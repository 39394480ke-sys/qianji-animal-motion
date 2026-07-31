import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = (
    REPOSITORY_ROOT / "experiments/2d_to_3d_cat/summarize_results.py"
)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_2d_experiment",
    SUMMARY_SCRIPT,
)
assert SUMMARY_SPEC is not None and SUMMARY_SPEC.loader is not None
SUMMARY_MODULE = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY_MODULE)


def test_39point_experiment_uses_the_verified_frozen_12x30_robot() -> None:
    script = (
        REPOSITORY_ROOT / "experiments/39point_vgt_cat/run_experiment.sh"
    ).read_text(encoding="utf-8")

    assert "qianji_animal_motion.frozen_initial_robot" in script
    assert "cat_hunyuan_qianji_robot_12x30.json" in script
    assert "cat_hunyuan_qianji_robot_12x30.manifest.json" in script
    assert 'cp "$OUTPUT_ROOT/initial_model/robot_base.json"' in script
    assert '"$GENERATOR" 3d-mesh' not in script


def test_39point_candidate_metadata_remains_valid_after_case_publish() -> None:
    script = (
        REPOSITORY_ROOT / "experiments/39point_vgt_cat/run_experiment.sh"
    ).read_text(encoding="utf-8")

    assert '--arg landmark_dir "../../scales/scale_${scale_code}/landmarks"' in script
    assert '--arg run_summary "reachability/run_summary.json"' in script
    assert 'landmark_dir="$selected_root/$(jq -r' in script
    assert '--arg landmark_dir "$scale_root/landmarks"' not in script


def test_2d_experiment_failure_does_not_publish_partial_case(
    tmp_path: Path,
) -> None:
    animal_root = tmp_path / "animal-data"
    mesh = animal_root / "data/processed/小猫_assimp.glb"
    video = animal_root / "data/processed/cat_walk_30fps_720p.mp4"
    predictions = (
        animal_root
        / "outputs/zero_shot/cat_walk"
        / (
            "cat_walk_30fps_720p_superanimal_quadruped_"
            "hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5"
        )
    )
    trajectory = (
        animal_root
        / "outputs/manual_correction/cat_walk/review_v2_migrated"
        / "keypoint_trajectory_2d_corrected.json"
    )
    mesh.parent.mkdir(parents=True)
    trajectory.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    video.write_bytes(b"video")
    predictions.parent.mkdir(parents=True)
    predictions.write_bytes(b"predictions")
    trajectory.write_text("{}\n", encoding="utf-8")
    lineage_sources = (
        animal_root
        / "outputs/manual_correction/cat_walk/cvat_export_v2"
        / "cvat_manifest.json",
        animal_root
        / "outputs/manual_correction/cat_walk/cvat_export_v2"
        / "annotations_migrated_from_review_v1_clean.xml",
        animal_root
        / "outputs/semantic_six/cat_walk_v2/keypoint_trajectory_2d.json",
        animal_root
        / "outputs/anchor_review/cat_walk_v2/identity_anchor_candidates.json",
    )
    for source in lineage_sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("{}\n", encoding="utf-8")

    qianji_root = tmp_path / "qianji"
    qianji_scripts = (
        qianji_root / "morph_generator/generate.py",
        qianji_root / "controller/check_keypoint_reachability.py",
        qianji_root / "controller/visualize_keypoint_motion.py",
    )
    for script in qianji_scripts:
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# fixture\n", encoding="utf-8")
    _initialize_repository(qianji_root)

    executable_root = tmp_path / "bin"
    executable_root.mkdir()
    mamba = executable_root / "mamba"
    mamba.write_text(
        "#!/bin/sh\n"
        'if [ "$5" = "-c" ]; then\n'
        "  shift 4\n"
        '  exec "$PYTHON_BIN" "$@"\n'
        "fi\n"
        "exit 23\n",
        encoding="utf-8",
    )
    mamba.chmod(0o755)

    output_root = tmp_path / "published-case"
    environment = {
        **os.environ,
        "ANIMAL_DATA_ROOT": str(animal_root),
        "QIANJI_ROOT": str(qianji_root),
        "OUTPUT_ROOT": str(output_root),
        "PYTHON_BIN": sys.executable,
        "ALLOW_DIRTY_EXPERIMENT": "1",
        "PATH": f"{executable_root}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(
        [
            "bash",
            str(
                REPOSITORY_ROOT
                / "experiments/2d_to_3d_cat/run_experiment.sh"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 23
    assert not output_root.exists()
    assert not list(tmp_path.glob(".published-case.case-staging.*"))


def test_39point_experiment_failure_does_not_publish_partial_case(
    tmp_path: Path,
) -> None:
    animal_root = tmp_path / "animal-data"
    sources = (
        animal_root / "data/processed/cat_walk_30fps_720p.mp4",
        animal_root
        / "outputs/zero_shot/cat_walk"
        / (
            "cat_walk_30fps_720p_superanimal_quadruped_"
            "hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5"
        ),
        animal_root / "data/processed/小猫_assimp.glb",
        animal_root
        / "outputs/manual_correction/cat_walk/review_v2_migrated"
        / "keypoint_trajectory_2d_corrected.json",
    )
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"deliberately-invalid-fixture")

    qianji_root = tmp_path / "qianji"
    qianji_scripts = (
        qianji_root / "morph_generator/generate.py",
        qianji_root / "controller/check_keypoint_reachability.py",
        qianji_root / "controller/optimize_morphology_for_motion.py",
        qianji_root / "controller/reachability.py",
        qianji_root / "controller/slide_control_adapter.py",
        qianji_root / "mujoco_builder/json2xml_v7_perrod.py",
        qianji_root / "mujoco_builder/add_scene_to_xml.py",
    )
    for script in qianji_scripts:
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# fixture\n", encoding="utf-8")

    output_root = tmp_path / "published-39-case"
    result = subprocess.run(
        [
            "bash",
            str(
                REPOSITORY_ROOT
                / "experiments/39point_vgt_cat/run_experiment.sh"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "ANIMAL_DATA_ROOT": str(animal_root),
            "QIANJI_ROOT": str(qianji_root),
            "OUTPUT_ROOT": str(output_root),
            "PYTHON_BIN": sys.executable,
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not output_root.exists()
    assert not list(tmp_path.glob(".published-39-case.case-staging.*"))


def test_2d_summary_requires_identical_scale_sources_and_relative_case_paths(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "case-staging"
    robot_path = output_root / "morphology/robot.json"
    rig_path = output_root / "rig_seed/rig_keypoints.json"
    rigidity_path = (
        output_root / "morphology/reports/rigidity_report_3d.json"
    )
    mesh = tmp_path / "mesh.glb"
    trajectory = tmp_path / "trajectory.json"
    mesh.write_bytes(b"mesh")
    trajectory.write_text("{}\n", encoding="utf-8")
    _write_json(
        robot_path,
        {"name": "cat", "sites": {"a": {}}, "rod_groups": []},
    )
    _write_json(
        rig_path,
        {"key_site_map": {"spine_front": "a"}},
    )
    _write_json(
        rigidity_path,
        {
            "rank": 1,
            "target_rank": 1,
            "node_degree_report": {"max_degree": 0},
        },
    )
    source_records = {
        "trajectory": {
            "path": str(trajectory.resolve()),
            "sha256": SUMMARY_MODULE._sha256(trajectory),
        },
        "robot": {
            "path": "morphology/robot.json",
            "sha256": SUMMARY_MODULE._sha256(robot_path),
        },
        "rig": {
            "path": "rig_seed/rig_keypoints.json",
            "sha256": SUMMARY_MODULE._sha256(rig_path),
        },
    }
    for scale in SUMMARY_MODULE.SCALES:
        scale_root = output_root / f"scale_{scale:.2f}"
        _write_json(
            scale_root / "lift_report.json",
            {
                "reference_frame": 0,
                "substitution_counts": {},
                "displacement_norm": {},
                "sources": source_records,
            },
        )
        _write_json(
            scale_root / "reachability/run_summary.json",
            {
                "summary": {
                    "status_counts": {
                        "feasible": 1,
                        "marginal": 0,
                        "unreachable": 0,
                    },
                    "marginal_or_feasible_fraction": 1.0,
                    "max_keypoint_error_m": 0.0,
                    "mean_keypoint_error_m": 0.0,
                }
            },
        )
        _write_json(
            scale_root / "preview/keypoint_motion_preview_report.json",
            {
                "n_frames": 1,
                "duration_s": 0.1,
                "outputs": {
                    "overview_png": "overview.png",
                    "three_view_png": "three.png",
                    "preview_mp4": "preview.mp4",
                },
            },
        )

    summary = SUMMARY_MODULE.summarize(output_root, mesh, trajectory)

    assert summary["sources"]["robot"]["path"] == "morphology/robot.json"
    assert summary["sources"]["rig"]["path"] == "rig_seed/rig_keypoints.json"
    assert summary["verified_scale_source_hashes"] == {
        label: item["sha256"] for label, item in source_records.items()
    }

    first_report = output_root / "scale_0.10/lift_report.json"
    payload = json.loads(first_report.read_text(encoding="utf-8"))
    payload["sources"]["trajectory"]["sha256"] = "0" * 64
    _write_json(first_report, payload)
    with pytest.raises(ValueError, match="scale lift source"):
        SUMMARY_MODULE.summarize(output_root, mesh, trajectory)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _initialize_repository(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "config",
            "user.email",
            "test@example.invalid",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "fixture"],
        check=True,
    )
