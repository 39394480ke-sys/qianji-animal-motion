"""Command-line orchestration for quadruped 2D-to-3D retargeting."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Mapping, Sequence

from qianji_animal_motion.artifact_io import publish_staged_files
from qianji_animal_motion.lift_3d import lift_trajectory


LINEAGE_SOURCE_LABELS = (
    "video",
    "predictions",
    "correction_manifest",
    "annotations",
    "baseline_trajectory",
    "identity_anchor_manifest",
)


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


def _recorded_path(path: Path, case_root: Path | None) -> str:
    if case_root is not None:
        try:
            return path.relative_to(case_root).as_posix()
        except ValueError:
            pass
    return str(path)


def _require_hash(
    payload: Mapping,
    field: str,
    expected: str,
    *,
    label: str,
) -> None:
    if payload.get(field) != expected:
        raise ValueError(f"{label} {field} does not match verified lineage")


def _validate_corrected_lineage(
    trajectory: dict,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
) -> None:
    if set(paths) != set(LINEAGE_SOURCE_LABELS):
        raise ValueError(
            "lineage paths must contain: " + ", ".join(LINEAGE_SOURCE_LABELS)
        )
    source = trajectory.get("source")
    anchor = trajectory.get("identity_anchor")
    correction = trajectory.get("manual_correction")
    if not all(isinstance(item, dict) for item in (source, anchor, correction)):
        raise ValueError("corrected trajectory lineage metadata is required")
    _require_hash(
        source,
        "video_sha256",
        hashes["video"],
        label="trajectory source video lineage",
    )
    _require_hash(
        source,
        "predictions_sha256",
        hashes["predictions"],
        label="trajectory source predictions lineage",
    )
    for field, source_label in (
        ("video_sha256", "video"),
        ("predictions_sha256", "predictions"),
        ("manifest_sha256", "identity_anchor_manifest"),
    ):
        _require_hash(
            anchor,
            field,
            hashes[source_label],
            label="trajectory identity anchor lineage",
        )
    for field, source_label in (
        ("manifest_sha256", "correction_manifest"),
        ("annotations_sha256", "annotations"),
        ("baseline_trajectory_sha256", "baseline_trajectory"),
    ):
        _require_hash(
            correction,
            field,
            hashes[source_label],
            label="trajectory manual correction lineage",
        )

    correction_manifest = _load_json(paths["correction_manifest"])
    if (
        correction_manifest.get("schema") != "qianji.cvat_correction_manifest"
        or correction_manifest.get("schema_version") != "1.0.0"
    ):
        raise ValueError("correction manifest contract is unsupported")
    for field, source_label in (
        ("video_sha256", "video"),
        ("predictions_sha256", "predictions"),
        ("baseline_trajectory_sha256", "baseline_trajectory"),
    ):
        _require_hash(
            correction_manifest,
            field,
            hashes[source_label],
            label="correction manifest lineage",
        )
    if correction_manifest.get("video") != trajectory.get("video"):
        raise ValueError("correction manifest video metadata differs from trajectory")

    anchor_manifest = _load_json(paths["identity_anchor_manifest"])
    if (
        anchor_manifest.get("schema") != "qianji.identity_anchor_candidates"
        or anchor_manifest.get("schema_version") != "1.1.0"
    ):
        raise ValueError("identity anchor manifest contract is unsupported")
    for field, source_label in (
        ("video", "video"),
        ("predictions", "predictions"),
    ):
        item = anchor_manifest.get(field)
        if not isinstance(item, dict):
            raise ValueError(f"identity anchor manifest {field} record is missing")
        _require_hash(
            item,
            "sha256",
            hashes[source_label],
            label=f"identity anchor manifest {field} lineage",
        )


def run_lift(
    *,
    trajectory_path: Path,
    robot_path: Path,
    rig_path: Path,
    output_dir: Path,
    reference_frame: int | None = None,
    motion_scale: float = 0.25,
    lineage_paths: Mapping[str, Path] | None = None,
    case_root: Path | None = None,
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
    resolved_lineage = (
        {
            label: Path(path).resolve()
            for label, path in lineage_paths.items()
        }
        if lineage_paths is not None
        else {}
    )
    if resolved_lineage and set(resolved_lineage) != set(LINEAGE_SOURCE_LABELS):
        raise ValueError(
            "lineage paths must contain: " + ", ".join(LINEAGE_SOURCE_LABELS)
        )
    for label, path in resolved_lineage.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} lineage source does not exist: {path}")

    output_dir = Path(output_dir).resolve()
    resolved_case_root = Path(case_root).resolve() if case_root is not None else None
    if resolved_case_root is not None:
        try:
            output_dir.relative_to(resolved_case_root)
        except ValueError as error:
            raise ValueError("output directory must be inside case_root") from error
    final = _output_paths(output_dir)
    for path in final:
        if path.exists():
            raise FileExistsError(
                f"output already exists: {path}; use a new output directory"
            )

    source_hashes = {
        label: _sha256(path) for label, path in source_paths.items()
    }
    lineage_hashes = {
        label: _sha256(path) for label, path in resolved_lineage.items()
    }
    trajectory = _load_json(source_paths["trajectory"])
    if resolved_lineage:
        _validate_corrected_lineage(
            trajectory,
            resolved_lineage,
            lineage_hashes,
        )
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
            "path": _recorded_path(path, resolved_case_root),
            "sha256": source_hashes[label],
        }
        for label, path in source_paths.items()
    }
    verified_lineage = {
        label: {
            "path": _recorded_path(path, resolved_case_root),
            "sha256": lineage_hashes[label],
        }
        for label, path in resolved_lineage.items()
    }
    binding = {**result.binding, "sources": sources}
    report = {
        **result.report,
        "sources": sources,
        "verified_lineage": verified_lineage,
    }

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
        for label, path in resolved_lineage.items():
            if _sha256(path) != lineage_hashes[label]:
                raise RuntimeError(f"{label} lineage source changed during lifting")
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
    parser.add_argument("--video", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--correction-manifest", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--baseline-trajectory", type=Path)
    parser.add_argument("--identity-anchor-manifest", type=Path)
    parser.add_argument("--case-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lineage_values = {
        "video": args.video,
        "predictions": args.predictions,
        "correction_manifest": args.correction_manifest,
        "annotations": args.annotations,
        "baseline_trajectory": args.baseline_trajectory,
        "identity_anchor_manifest": args.identity_anchor_manifest,
    }
    present = {label for label, path in lineage_values.items() if path is not None}
    if present and present != set(LINEAGE_SOURCE_LABELS):
        missing = sorted(set(LINEAGE_SOURCE_LABELS) - present)
        raise ValueError(
            "all corrected-lineage options are required together; missing: "
            + ", ".join(missing)
        )
    outputs = run_lift(
        trajectory_path=args.trajectory,
        robot_path=args.robot_json,
        rig_path=args.rig,
        output_dir=args.output,
        reference_frame=args.reference_frame,
        motion_scale=args.motion_scale,
        lineage_paths=lineage_values if present else None,
        case_root=args.case_root,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
