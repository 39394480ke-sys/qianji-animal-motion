import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from qianji_animal_motion.experiment_provenance import build_provenance
from qianji_animal_motion.experiment_provenance import (
    verify_unchanged_experiment_state,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(path: Path, content: str) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "tracked.py").write_text(content, encoding="utf-8")
    _git(path, "add", "tracked.py")
    _git(path, "commit", "-qm", "fixture")
    return path


def test_provenance_records_revisions_scripts_environment_and_dirty_state(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "repository", "repo-v1\n")
    qianji = _repository(tmp_path / "qianji", "qianji-v1\n")
    script = qianji / "tracked.py"

    clean = build_provenance(
        repository_root=repository,
        qianji_root=qianji,
        script_paths=[script],
        package_names=["pytest", "package-that-does-not-exist"],
        invocation=["run_experiment.sh", "--fixture"],
        qianji_python_command=[sys.executable],
    )

    assert clean["schema"] == "qianji.experiment_provenance"
    assert clean["repository"]["commit"] == _git(repository, "rev-parse", "HEAD")
    assert clean["repository"]["dirty"] is False
    assert clean["qianji"]["commit"] == _git(qianji, "rev-parse", "HEAD")
    assert clean["qianji"]["dirty"] is False
    assert clean["scripts"] == [
        {
            "path": str(script.resolve()),
            "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        }
    ]
    assert clean["environment"]["python_version"]
    assert clean["environment"]["python_executable"]
    assert clean["environment"]["packages"]["pytest"]
    assert clean["qianji_environment"]["python_executable"] == sys.executable
    assert clean["qianji_environment"]["packages"]["pytest"]
    assert clean["invocation"]["argv"] == [
        "run_experiment.sh",
        "--fixture",
    ]
    assert clean["invocation"]["working_directory"]
    assert (
        clean["environment"]["packages"]["package-that-does-not-exist"]
        is None
    )

    script.write_text("qianji-dirty\n", encoding="utf-8")
    dirty = build_provenance(
        repository_root=repository,
        qianji_root=qianji,
        script_paths=[script],
        package_names=[],
    )

    assert dirty["qianji"]["dirty"] is True
    assert dirty["qianji"]["status_porcelain"] == [" M tracked.py"]
    assert (
        dirty["qianji"]["worktree_state_sha256"]
        != clean["qianji"]["worktree_state_sha256"]
    )


def test_provenance_changes_when_qianji_revision_changes(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository", "repo-v1\n")
    qianji = _repository(tmp_path / "qianji", "qianji-v1\n")

    before = build_provenance(
        repository_root=repository,
        qianji_root=qianji,
        script_paths=[qianji / "tracked.py"],
        package_names=[],
    )
    (qianji / "tracked.py").write_text("qianji-v2\n", encoding="utf-8")
    _git(qianji, "add", "tracked.py")
    _git(qianji, "commit", "-qm", "second")
    after = build_provenance(
        repository_root=repository,
        qianji_root=qianji,
        script_paths=[qianji / "tracked.py"],
        package_names=[],
    )

    assert before["qianji"]["commit"] != after["qianji"]["commit"]
    assert before != after


def test_provenance_locks_inputs_tools_and_repositories_for_whole_run(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "repository", "repo-v1\n")
    qianji = _repository(tmp_path / "qianji", "qianji-v1\n")
    source = tmp_path / "trajectory.json"
    source.write_text("{}\n", encoding="utf-8")
    arguments = {
        "repository_root": repository,
        "qianji_root": qianji,
        "script_paths": [qianji / "tracked.py"],
        "input_paths": [source],
        "package_names": [],
        "invocation": ["bash", "run_experiment.sh"],
        "working_directory": repository,
        "qianji_python_command": [sys.executable],
    }

    start = build_provenance(**arguments)
    end = build_provenance(**arguments)

    verify_unchanged_experiment_state(start, end, require_clean=True)
    assert end["invocation"] == {
        "argv": ["bash", "run_experiment.sh"],
        "working_directory": str(repository.resolve()),
    }
    assert end["inputs"] == [
        {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    assert end["qianji_environment"]["python_executable"] == sys.executable

    source.write_text('{"changed": true}\n', encoding="utf-8")
    changed = build_provenance(**arguments)
    with pytest.raises(ValueError, match="input files changed"):
        verify_unchanged_experiment_state(start, changed, require_clean=True)
