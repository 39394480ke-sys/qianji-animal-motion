import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
