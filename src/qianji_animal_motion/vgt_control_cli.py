"""Prepare bbox and motion-informed VGT control candidates without mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from qianji_animal_motion.artifact_io import publish_staged_files
from qianji_animal_motion.vgt_control import (
    apply_contraction_range,
    build_motion_informed_rig,
    build_target_control_motion,
    build_vgt_control_map,
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
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fraction_suffix(fraction: float) -> str:
    value = round(fraction * 100)
    if abs(value / 100.0 - fraction) > 1e-9:
        raise ValueError("contraction fractions must use one-percent increments")
    return f"{value:03d}"


def run_prepare_vgt_control(
    *,
    robot_path: Path,
    bbox_rig_path: Path,
    neutral_landmarks_path: Path,
    motion_39_path: Path,
    output_dir: Path,
    contraction_fractions: Sequence[float] = (0.0, 0.1),
) -> tuple[Path, ...]:
    sources = {
        "robot": Path(robot_path).resolve(),
        "bbox_rig": Path(bbox_rig_path).resolve(),
        "neutral": Path(neutral_landmarks_path).resolve(),
        "motion": Path(motion_39_path).resolve(),
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} source does not exist: {path}")
    fractions = tuple(float(value) for value in contraction_fractions)
    suffixes = tuple(_fraction_suffix(value) for value in fractions)
    if not fractions or len(set(suffixes)) != len(suffixes):
        raise ValueError("contraction fractions must be non-empty and unique")

    output_dir = Path(output_dir).resolve()
    names = [
        "rig_bbox.json",
        "rig_motion_informed.json",
        "control_map_bbox.json",
        "control_map_motion_informed.json",
        "target_control_motion_bbox.json",
        "target_control_motion_motion_informed.json",
        "control_report_bbox.json",
        "control_report_motion_informed.json",
        *[f"robot_contraction_{suffix}.json" for suffix in suffixes],
        "vgt_control_candidates_manifest.json",
    ]
    final_paths = tuple(output_dir / name for name in names)
    for path in final_paths:
        if path.exists():
            raise FileExistsError(
                f"output already exists: {path}; use a new output directory"
            )

    hashes = {label: _sha256(path) for label, path in sources.items()}
    robot = _load_json(sources["robot"])
    bbox_rig = _load_json(sources["bbox_rig"])
    neutral = _load_json(sources["neutral"])
    motion = _load_json(sources["motion"])
    motion_rig = build_motion_informed_rig(robot, neutral)
    artifacts: dict[str, dict] = {
        "rig_bbox.json": bbox_rig,
        "rig_motion_informed.json": motion_rig,
    }
    for label, rig in (("bbox", bbox_rig), ("motion_informed", motion_rig)):
        control_map = build_vgt_control_map(robot, rig, neutral)
        target, report = build_target_control_motion(
            motion,
            neutral,
            robot,
            control_map,
        )
        artifacts[f"control_map_{label}.json"] = control_map
        artifacts[f"target_control_motion_{label}.json"] = target
        artifacts[f"control_report_{label}.json"] = report
    for fraction, suffix in zip(fractions, suffixes, strict=True):
        artifacts[f"robot_contraction_{suffix}.json"] = apply_contraction_range(
            robot,
            fraction,
        )
    artifacts["vgt_control_candidates_manifest.json"] = {
        "schema": "qianji.vgt_control_candidates_manifest",
        "schema_version": "0.1.0",
        "source_paths": {
            label: str(path)
            for label, path in sources.items()
        },
        "source_hashes": hashes,
        "rig_variants": ["bbox", "motion_informed"],
        "contraction_fractions": list(fractions),
        "files": names[:-1],
        "input_mutation": False,
        "qianji_mutation": False,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}.control-staging-",
        dir=output_dir.parent,
    ) as temporary:
        staged_root = Path(temporary)
        staged_paths = tuple(staged_root / name for name in names)
        for path in staged_paths:
            _write_json(path, artifacts[path.name])
        for label, path in sources.items():
            if _sha256(path) != hashes[label]:
                raise RuntimeError(f"{label} source changed during preparation")
        publish_staged_files(
            staged_paths,
            final_paths,
            overwrite=False,
            conflict_hint="use a new output directory",
        )
    return final_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-json", type=Path, required=True)
    parser.add_argument("--bbox-rig", type=Path, required=True)
    parser.add_argument("--neutral-landmarks", type=Path, required=True)
    parser.add_argument("--motion-39", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contraction-fractions",
        type=float,
        nargs="+",
        default=(0.0, 0.1),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = run_prepare_vgt_control(
        robot_path=args.robot_json,
        bbox_rig_path=args.bbox_rig,
        neutral_landmarks_path=args.neutral_landmarks,
        motion_39_path=args.motion_39,
        output_dir=args.output,
        contraction_fractions=args.contraction_fractions,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
