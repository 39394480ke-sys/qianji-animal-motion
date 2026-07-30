import hashlib
import subprocess
from pathlib import Path

from qianji_animal_motion.experiment_provenance import build_provenance


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
