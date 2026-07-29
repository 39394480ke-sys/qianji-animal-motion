"""Command-line orchestration for quadruped 2D-to-3D retargeting."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Sequence

from qianji_animal_motion.artifact_io import publish_staged_files
from qianji_animal_motion.lift_3d import lift_trajectory


@dataclass(frozen=True)
class LiftOutputPaths:
    motion: Path
    binding: Path
    report: Path

    def __iter__(self) -> Iterator[Path]:
        return iter((self.motion, self.binding, self.report))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON source: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _output_paths(output_dir: Path) -> LiftOutputPaths:
    return LiftOutputPaths(
        motion=output_dir / "keypoint_motion.json",
        binding=output_dir / "mesh_binding.json",
        report=output_dir / "lift_report.json",
    )


def run_lift(
    *,
    trajectory_path: Path,
    robot_path: Path,
    rig_path: Path,
    output_dir: Path,
    reference_frame: int | None = None,
    motion_scale: float = 0.25,
) -> LiftOutputPaths:
    """Create one complete, non-overwriting lifted-motion artifact set."""
    source_paths = {
        "trajectory": Path(trajectory_path).resolve(),
        "robot": Path(robot_path).resolve(),
        "rig": Path(rig_path).resolve(),
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} source does not exist: {path}")

    output_dir = Path(output_dir).resolve()
    final = _output_paths(output_dir)
    for path in final:
        if path.exists():
            raise FileExistsError(
                f"output already exists: {path}; use a new output directory"
            )

    source_hashes = {
        label: _sha256(path) for label, path in source_paths.items()
    }
    trajectory = _load_json(source_paths["trajectory"])
    robot = _load_json(source_paths["robot"])
    rig = _load_json(source_paths["rig"])
    result = lift_trajectory(
        trajectory,
        robot,
        rig,
        reference_frame=reference_frame,
        motion_scale=motion_scale,
    )

    sources = {
        label: {
            "path": str(path),
            "sha256": source_hashes[label],
        }
        for label, path in source_paths.items()
    }
    binding = {**result.binding, "sources": sources}
    report = {**result.report, "sources": sources}

    staging_parent = output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=staging_parent,
    ) as temporary:
        staged = _output_paths(Path(temporary))
        _write_json(staged.motion, result.motion)
        _write_json(staged.binding, binding)
        _write_json(staged.report, report)
        for label, path in source_paths.items():
            if _sha256(path) != source_hashes[label]:
                raise RuntimeError(f"{label} source changed during lifting")
        publish_staged_files(
            tuple(staged),
            tuple(final),
            overwrite=False,
            conflict_hint="use a new output directory",
        )
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget six body-relative 2D quadruped points onto a QianJi rig."
        )
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--robot-json", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int)
    parser.add_argument("--motion-scale", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = run_lift(
        trajectory_path=args.trajectory,
        robot_path=args.robot_json,
        rig_path=args.rig,
        output_dir=args.output,
        reference_frame=args.reference_frame,
        motion_scale=args.motion_scale,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
