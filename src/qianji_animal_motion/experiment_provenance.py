"""Record reproducible repository, script, and Python environment provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
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


def _python_environment(
    package_names: Sequence[str],
    command: Sequence[str] | None = None,
) -> dict:
    names = sorted(set(package_names))
    if command is None:
        packages = {}
        for name in names:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "packages": packages,
        }
    if not command:
        raise ValueError("Python environment command cannot be empty")
    source = (
        "import importlib.metadata,json,os,platform,sys;"
        f"names={names!r};"
        "packages={};"
        "\nfor name in names:\n"
        " try: packages[name]=importlib.metadata.version(name)\n"
        " except importlib.metadata.PackageNotFoundError: packages[name]=None\n"
        "print(json.dumps({"
        "'conda_default_env':os.environ.get('CONDA_DEFAULT_ENV'),"
        "'conda_prefix':os.environ.get('CONDA_PREFIX'),"
        "'python_executable':sys.executable,"
        "'python_version':platform.python_version(),"
        "'packages':packages}))"
    )
    try:
        completed = subprocess.run(
            [*command, "-c", source],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(
            "could not inspect the QianJi Python environment"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("QianJi Python environment output is malformed")
    return payload


def build_provenance(
    *,
    repository_root: Path,
    qianji_root: Path,
    script_paths: Sequence[Path],
    input_paths: Sequence[Path] = (),
    package_names: Sequence[str] = DEFAULT_PACKAGES,
    invocation: Sequence[str] | None = None,
    working_directory: Path | None = None,
    qianji_python_command: Sequence[str] | None = None,
) -> dict:
    """Build a deterministic provenance payload for one experiment run."""
    scripts = []
    for source in script_paths:
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"provenance script does not exist: {path}")
        scripts.append({"path": str(path), "sha256": _sha256(path)})
    inputs = []
    for source in input_paths:
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"provenance input does not exist: {path}")
        inputs.append({"path": str(path), "sha256": _sha256(path)})

    return {
        "schema": "qianji.experiment_provenance",
        "schema_version": "1.0.0",
        "repository": _repository_state(Path(repository_root)),
        "qianji": _repository_state(Path(qianji_root)),
        "scripts": scripts,
        "inputs": inputs,
        "invocation": {
            "argv": list(sys.argv if invocation is None else invocation),
            "working_directory": str(
                Path(
                    os.getcwd()
                    if working_directory is None
                    else working_directory
                ).resolve()
            ),
        },
        "environment": _python_environment(package_names),
        "qianji_environment": (
            None
            if qianji_python_command is None
            else _python_environment(package_names, qianji_python_command)
        ),
    }


def verify_unchanged_experiment_state(
    start: dict,
    end: dict,
    *,
    require_clean: bool,
) -> None:
    """Reject a run whose repositories, tools, inputs, or Python environment changed."""
    for label in ("repository", "qianji"):
        before = start.get(label)
        after = end.get(label)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"{label} provenance is missing")
        if require_clean and (
            before.get("dirty") is not False or after.get("dirty") is not False
        ):
            raise ValueError(f"{label} must remain clean for a formal experiment")
        if before != after:
            raise ValueError(f"{label} state changed during the experiment")
    if start.get("scripts") != end.get("scripts"):
        raise ValueError("tool scripts changed during the experiment")
    if start.get("inputs") != end.get("inputs"):
        raise ValueError("input files changed during the experiment")
    if start.get("environment") != end.get("environment"):
        raise ValueError("Python environment changed during the experiment")
    if start.get("qianji_environment") != end.get("qianji_environment"):
        raise ValueError("QianJi Python environment changed during the experiment")
    if start.get("invocation") != end.get("invocation"):
        raise ValueError("experiment invocation changed during the experiment")


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
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--package", action="append")
    parser.add_argument("--invocation-arg", action="append")
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--qianji-python-command")
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_provenance(
        repository_root=args.repository_root,
        qianji_root=args.qianji_root,
        script_paths=args.script,
        input_paths=args.input,
        package_names=(
            DEFAULT_PACKAGES if args.package is None else args.package
        ),
        invocation=args.invocation_arg,
        working_directory=args.working_directory,
        qianji_python_command=(
            None
            if args.qianji_python_command is None
            else shlex.split(args.qianji_python_command)
        ),
    )
    if args.verify_against is not None:
        baseline = json.loads(args.verify_against.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise ValueError("baseline provenance must contain a JSON object")
        verify_unchanged_experiment_state(
            baseline,
            payload,
            require_clean=args.require_clean,
        )
    elif args.require_clean:
        verify_unchanged_experiment_state(
            payload,
            payload,
            require_clean=True,
        )
    write_provenance(args.output, payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
