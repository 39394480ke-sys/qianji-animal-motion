"""Record reproducible repository, script, and Python environment provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence


DEFAULT_PACKAGES = (
    "mujoco",
    "numpy",
    "pandas",
    "scipy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(
    repository: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def _repository_state(repository: Path) -> dict:
    repository = repository.resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"repository does not exist: {repository}")
    try:
        commit = str(_git(repository, "rev-parse", "HEAD")).strip()
        status_text = str(
            _git(
                repository,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"not a Git repository: {repository}") from error

    status = status_text.splitlines()
    state_digest = hashlib.sha256()
    state_digest.update(
        _git(repository, "diff", "--binary", "HEAD", text=False)
    )
    untracked = _git(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        text=False,
    )
    assert isinstance(untracked, bytes)
    for raw_relative in sorted(filter(None, untracked.split(b"\0"))):
        relative = os.fsdecode(raw_relative)
        path = repository / relative
        state_digest.update(raw_relative)
        state_digest.update(b"\0")
        if path.is_file():
            state_digest.update(path.read_bytes())
        state_digest.update(b"\0")

    return {
        "path": str(repository),
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status,
        "worktree_state_sha256": state_digest.hexdigest(),
    }


def build_provenance(
    *,
    repository_root: Path,
    qianji_root: Path,
    script_paths: Sequence[Path],
    package_names: Sequence[str] = DEFAULT_PACKAGES,
) -> dict:
    """Build a deterministic provenance payload for one experiment run."""
    scripts = []
    for source in script_paths:
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"provenance script does not exist: {path}")
        scripts.append({"path": str(path), "sha256": _sha256(path)})

    packages = {}
    for name in sorted(set(package_names)):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "schema": "qianji.experiment_provenance",
        "schema_version": "1.0.0",
        "repository": _repository_state(Path(repository_root)),
        "qianji": _repository_state(Path(qianji_root)),
        "scripts": scripts,
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "packages": packages,
        },
    }


def write_provenance(path: Path, payload: dict) -> None:
    """Publish a provenance JSON file without overwriting an existing file."""
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"provenance output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.staging-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            raise FileExistsError(f"provenance output already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--qianji-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, action="append", default=[])
    parser.add_argument("--package", action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_provenance(
        repository_root=args.repository_root,
        qianji_root=args.qianji_root,
        script_paths=args.script,
        package_names=(
            DEFAULT_PACKAGES if args.package is None else args.package
        ),
    )
    write_provenance(args.output, payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
