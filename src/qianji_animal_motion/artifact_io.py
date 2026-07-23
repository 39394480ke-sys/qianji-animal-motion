"""Helpers for publishing related artifacts as one recoverable operation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_staged_files(
    staged_paths: Sequence[Path],
    final_paths: Sequence[Path],
    *,
    overwrite: bool,
    conflict_hint: str,
) -> None:
    """Move a complete staged file set into place, rolling back on failure."""
    staged = tuple(Path(path) for path in staged_paths)
    final = tuple(Path(path) for path in final_paths)
    if not staged or len(staged) != len(final):
        raise ValueError("staged and final artifact lists must have equal length")
    missing = [path for path in staged if not path.is_file()]
    if missing:
        raise RuntimeError(f"staged output is missing: {missing[0]}")
    expected_hashes = tuple(_sha256(path) for path in staged)
    existing = [path for path in final if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output already exists: {existing[0]}; {conflict_hint}")

    output_dirs = {path.parent for path in final}
    preexisting_dirs = {path: path.exists() for path in output_dirs}
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = staged[0].parent / ".backup"
    backup_dir.mkdir()
    backups: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for index, final_path in enumerate(final):
            if final_path.exists():
                backup_path = backup_dir / f"{index}-{final_path.name}"
                final_path.replace(backup_path)
                backups.append((backup_path, final_path))
        for staged_path, final_path in zip(staged, final, strict=True):
            staged_path.replace(final_path)
            published.append((final_path, staged_path))
        for final_path, expected_hash in zip(
            final,
            expected_hashes,
            strict=True,
        ):
            if not final_path.is_file():
                raise RuntimeError(
                    f"output is missing after publishing: {final_path}"
                )
            if _sha256(final_path) != expected_hash:
                raise RuntimeError(
                    f"output changed while publishing: {final_path}"
                )
    except Exception as exc:
        rollback_errors: list[OSError] = []
        for final_path, staged_path in reversed(published):
            if final_path.exists():
                try:
                    final_path.replace(staged_path)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)
        for backup_path, final_path in reversed(backups):
            if backup_path.exists():
                try:
                    backup_path.replace(final_path)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)
        for output_dir, existed in preexisting_dirs.items():
            if not existed:
                try:
                    output_dir.rmdir()
                except OSError:
                    pass
        if rollback_errors:
            raise RuntimeError(
                "artifact publishing failed and rollback was incomplete"
            ) from exc
        raise
