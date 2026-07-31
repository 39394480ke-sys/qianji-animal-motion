"""CLI for publishing neutral and moving 39-point 2.5D landmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Sequence

from qianji_animal_motion.artifact_io import publish_staged_files
from qianji_animal_motion.lift_39 import (
    build_neutral_landmarks_39,
    lift_39point_trajectory,
)


@dataclass(frozen=True)
class Lift39Paths:
    neutral: Path
    motion: Path
    report: Path

    def __iter__(self) -> Iterator[Path]:
        return iter((self.neutral, self.motion, self.report))


def _paths(root: Path) -> Lift39Paths:
    return Lift39Paths(
        neutral=root / "neutral_landmarks_39.json",
        motion=root / "keypoint_motion_3d_39.json",
        report=root / "lift_39_report.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def _recorded_path(path: Path, case_root: Path | None) -> str:
    if case_root is None:
        return str(path)
    try:
        return path.relative_to(case_root).as_posix()
    except ValueError:
        return str(path)


def run_lift_39(
    *,
    trajectory_39_path: Path,
    corrected_spine_path: Path,
    robot_path: Path,
    rig_path: Path,
    output_dir: Path,
    reference_frame: int = 152,
    motion_scale: float = 0.1,
    case_root: Path | None = None,
) -> Lift39Paths:
    sources = {
        "trajectory_39": Path(trajectory_39_path).resolve(),
        "corrected_spine": Path(corrected_spine_path).resolve(),
        "robot": Path(robot_path).resolve(),
        "rig": Path(rig_path).resolve(),
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} source does not exist: {path}")
    output_dir = Path(output_dir).resolve()
    resolved_case_root = (
        None if case_root is None else Path(case_root).resolve()
    )
    if resolved_case_root is not None and not resolved_case_root.is_dir():
        raise FileNotFoundError(
            f"case root does not exist: {resolved_case_root}"
        )
    final = _paths(output_dir)
    for path in final:
        if path.exists():
            raise FileExistsError(
                f"output already exists: {path}; use a new output directory"
            )
    hashes = {label: _sha256(path) for label, path in sources.items()}
    trajectory = _load_json(sources["trajectory_39"])
    corrected = _load_json(sources["corrected_spine"])
    lineage = trajectory.get("source_lineage")
    if (
        not isinstance(lineage, dict)
        or lineage.get("corrected_trajectory_sha256")
        != hashes["corrected_spine"]
    ):
        raise ValueError(
            "trajectory corrected_trajectory_sha256 does not match "
            "the corrected spine source"
        )
    robot = _load_json(sources["robot"])
    rig = _load_json(sources["rig"])
    neutral = build_neutral_landmarks_39(
        trajectory,
        corrected,
        robot,
        rig,
        reference_frame=reference_frame,
    )
    source_records = {
        label: {
            "path": _recorded_path(path, resolved_case_root),
            "sha256": hashes[label],
        }
        for label, path in sources.items()
    }
    neutral = {**neutral, "sources": source_records}
    result = lift_39point_trajectory(
        trajectory,
        corrected,
        neutral,
        motion_scale=motion_scale,
    )
    report = {**result.report, "sources": source_records}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}.lift39-staging-",
        dir=output_dir.parent,
    ) as temporary:
        staged = _paths(Path(temporary))
        _write_json(staged.neutral, result.neutral_landmarks)
        _write_json(staged.motion, result.motion)
        _write_json(staged.report, report)
        for label, path in sources.items():
            if _sha256(path) != hashes[label]:
                raise RuntimeError(f"{label} source changed during lifting")
        publish_staged_files(
            tuple(staged),
            tuple(final),
            overwrite=False,
            conflict_hint="use a new output directory",
        )
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-39", type=Path, required=True)
    parser.add_argument("--corrected-spine", type=Path, required=True)
    parser.add_argument("--robot-json", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=152)
    parser.add_argument("--motion-scale", type=float, default=0.1)
    parser.add_argument("--case-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = run_lift_39(
        trajectory_39_path=args.trajectory_39,
        corrected_spine_path=args.corrected_spine,
        robot_path=args.robot_json,
        rig_path=args.rig,
        output_dir=args.output,
        reference_frame=args.reference_frame,
        motion_scale=args.motion_scale,
        case_root=args.case_root,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
