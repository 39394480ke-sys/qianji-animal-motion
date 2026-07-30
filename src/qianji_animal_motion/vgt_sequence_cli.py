"""Validate, package, hash, and render a selected QianJi VGT sequence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np

from qianji_animal_motion.artifact_io import publish_staged_files
from qianji_animal_motion.vgt_render import render_vgt_motion
from qianji_animal_motion.vgt_sequence import (
    load_and_validate_vgt_sequence,
    validate_control_pair_against_sequence,
    validate_render_rate,
)


_OUTPUT_NAMES = (
    "vgt_motion.npz",
    "vgt_motion_manifest.json",
    "vgt_three_view.png",
    "vgt_isometric.png",
    "vgt_motion_30fps.mp4",
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


def _case_relative(path: Path, case_root: Path) -> str:
    try:
        return path.resolve().relative_to(case_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"packaged artifact must be inside the case root: {path}"
        ) from error


def _relative_selected_candidate(selected: dict, case_root: Path) -> dict:
    output = copy.deepcopy(selected)
    candidate_root = output.get("candidate_root")
    if isinstance(candidate_root, str) and Path(candidate_root).is_absolute():
        output["candidate_root"] = _case_relative(
            Path(candidate_root),
            case_root,
        )
    artifacts = output.get("artifacts")
    if isinstance(artifacts, dict):
        for item in artifacts.values():
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            if isinstance(raw_path, str) and Path(raw_path).is_absolute():
                item["path"] = _case_relative(Path(raw_path), case_root)
    return output


def run_package_vgt_motion(
    *,
    source_npz_path: Path,
    robot_path: Path,
    rig_path: Path,
    desired_motion_path: Path,
    projected_motion_path: Path,
    selected_candidate_path: Path,
    output_dir: Path,
    expected_frames: int = 272,
    expected_sites: int = 12,
    expected_rods: int = 30,
    width: int = 1246,
    height: int = 720,
    fps: float = 30.0,
) -> tuple[Path, ...]:
    sources = {
        "source_npz": Path(source_npz_path).resolve(),
        "robot": Path(robot_path).resolve(),
        "rig": Path(rig_path).resolve(),
        "desired": Path(desired_motion_path).resolve(),
        "projected": Path(projected_motion_path).resolve(),
        "selected_candidate": Path(selected_candidate_path).resolve(),
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} source does not exist: {path}")
    if sources["desired"] == sources["projected"]:
        raise ValueError("desired and projected motions must be different files")
    source_hashes = {label: _sha256(path) for label, path in sources.items()}
    if source_hashes["desired"] == source_hashes["projected"]:
        raise ValueError("desired and projected motions must have different content")

    output_dir = Path(output_dir).resolve()
    case_root = output_dir.parent
    final_paths = tuple(output_dir / name for name in _OUTPUT_NAMES)
    for path in final_paths:
        if path.exists():
            raise FileExistsError(
                f"output already exists: {path}; use a new output directory"
            )
    robot = _load_json(sources["robot"])
    rig = _load_json(sources["rig"])
    desired = _load_json(sources["desired"])
    projected = _load_json(sources["projected"])
    selected = _load_json(sources["selected_candidate"])
    sequence = load_and_validate_vgt_sequence(
        sources["source_npz"],
        robot,
        expected_frames=expected_frames,
        expected_sites=expected_sites,
        expected_rods=expected_rods,
    )
    validate_render_rate(sequence, fps)
    validate_control_pair_against_sequence(
        desired,
        projected,
        sequence,
        rig,
        expected_frames=expected_frames,
        expected_fps=fps,
    )
    if selected.get("eligible") is not True:
        raise ValueError("selected candidate must be eligible")
    selected_artifacts = selected.get("artifacts")
    if not isinstance(selected_artifacts, dict):
        raise ValueError("selected candidate artifacts are required")
    selected_sources = {
        "robot": "robot",
        "rig": "rig",
        "desired_motion": "desired",
        "projected_motion": "projected",
        "site_npz": "source_npz",
    }
    for artifact_label, source_label in selected_sources.items():
        artifact = selected_artifacts.get(artifact_label)
        if (
            not isinstance(artifact, dict)
            or artifact.get("sha256") != source_hashes[source_label]
        ):
            raise ValueError(
                f"selected candidate {artifact_label} hash does not match input"
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}.vgt-staging-",
        dir=output_dir.parent,
    ) as temporary:
        staged_root = Path(temporary)
        staged_npz = staged_root / "vgt_motion.npz"
        np.savez_compressed(
            staged_npz,
            site_names=np.asarray(sequence.site_names),
            times=sequence.times,
            positions=sequence.positions,
        )
        render_report = render_vgt_motion(
            sequence,
            robot,
            staged_root,
            width=width,
            height=height,
            fps=fps,
        )
        rods = [
            {
                "name": item["name"],
                "site1": item["site1"],
                "site2": item["site2"],
            }
            for item in robot["rod_groups"]
        ]
        manifest = {
            "schema": "qianji.vgt_motion_manifest",
            "schema_version": "0.1.0",
            "selected_candidate": _relative_selected_candidate(
                selected,
                case_root,
            ),
            "frame_count": expected_frames,
            "fps": float(fps),
            "time_range": [
                float(sequence.times[0]),
                float(sequence.times[-1]),
            ],
            "site_count": expected_sites,
            "site_names": list(sequence.site_names),
            "rod_count": expected_rods,
            "rods": rods,
            "times_shape": list(sequence.times.shape),
            "positions_shape": list(sequence.positions.shape),
            "vgt_motion": {
                "path": _case_relative(
                    output_dir / "vgt_motion.npz",
                    case_root,
                ),
                "sha256": _sha256(staged_npz),
                "source_path": _case_relative(
                    sources["source_npz"],
                    case_root,
                ),
                "source_sha256": source_hashes["source_npz"],
            },
            "robot": {
                "path": _case_relative(sources["robot"], case_root),
                "sha256": source_hashes["robot"],
            },
            "rig": {
                "path": _case_relative(sources["rig"], case_root),
                "sha256": source_hashes["rig"],
                "key_site_map": rig.get("key_site_map"),
            },
            "desired_control_motion": {
                "path": _case_relative(sources["desired"], case_root),
                "sha256": source_hashes["desired"],
                "schema": desired.get("schema"),
            },
            "projected_control_motion": {
                "path": _case_relative(sources["projected"], case_root),
                "sha256": source_hashes["projected"],
                "schema": projected.get("schema"),
            },
            "render": {
                **render_report,
                "three_view": _case_relative(
                    output_dir / "vgt_three_view.png",
                    case_root,
                ),
                "isometric": _case_relative(
                    output_dir / "vgt_isometric.png",
                    case_root,
                ),
                "video": _case_relative(
                    output_dir / "vgt_motion_30fps.mp4",
                    case_root,
                ),
            },
            "scientific_limits": {
                "metric_depth_observed": False,
                "camera_calibrated": False,
                "global_translation_preserved": False,
                "dynamics_simulated": False,
                "mesh_vertices_deformed": False,
            },
        }
        staged_manifest = staged_root / "vgt_motion_manifest.json"
        _write_json(staged_manifest, manifest)
        for label, path in sources.items():
            if _sha256(path) != source_hashes[label]:
                raise RuntimeError(f"{label} source changed during packaging")
        staged_paths = tuple(staged_root / name for name in _OUTPUT_NAMES)
        publish_staged_files(
            staged_paths,
            final_paths,
            overwrite=False,
            conflict_hint="use a new output directory",
        )
    return final_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-npz", type=Path, required=True)
    parser.add_argument("--robot-json", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--desired-control-motion", type=Path, required=True)
    parser.add_argument("--projected-control-motion", type=Path, required=True)
    parser.add_argument("--selected-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=272)
    parser.add_argument("--expected-sites", type=int, default=12)
    parser.add_argument("--expected-rods", type=int, default=30)
    parser.add_argument("--width", type=int, default=1246)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = run_package_vgt_motion(
        source_npz_path=args.site_npz,
        robot_path=args.robot_json,
        rig_path=args.rig,
        desired_motion_path=args.desired_control_motion,
        projected_motion_path=args.projected_control_motion,
        selected_candidate_path=args.selected_candidate,
        output_dir=args.output,
        expected_frames=args.expected_frames,
        expected_sites=args.expected_sites,
        expected_rods=args.expected_rods,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
