import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
    trajectory = (
        animal_root
        / "outputs/manual_correction/cat_walk/review_v2_migrated"
        / "keypoint_trajectory_2d_corrected.json"
    )
    mesh.parent.mkdir(parents=True)
    trajectory.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    trajectory.write_text("{}\n", encoding="utf-8")

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
    mamba.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    mamba.chmod(0o755)

    output_root = tmp_path / "published-case"
    environment = {
        **os.environ,
        "ANIMAL_DATA_ROOT": str(animal_root),
        "QIANJI_ROOT": str(qianji_root),
        "OUTPUT_ROOT": str(output_root),
        "PYTHON_BIN": sys.executable,
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
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not output_root.exists()
    assert not list(tmp_path.glob(".published-39-case.case-staging.*"))


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
