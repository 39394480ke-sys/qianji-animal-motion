"""CLI for publishing complete SuperAnimal 39-point observation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Sequence

import cv2
import pandas as pd

from qianji_animal_motion.artifact_io import publish_staged_files
from qianji_animal_motion.keypoints_39 import (
    SUPERANIMAL_QUADRUPED_39,
    Observation39Result,
    build_39point_observation,
)
from qianji_animal_motion.semantic_cli import probe_video


_EDGES = (
    ("nose", "upper_jaw"),
    ("upper_jaw", "lower_jaw"),
    ("right_eye", "right_earbase"),
    ("right_earbase", "right_earend"),
    ("left_eye", "left_earbase"),
    ("left_earbase", "left_earend"),
    ("neck_base", "neck_end"),
    ("neck_end", "back_base"),
    ("back_base", "back_middle"),
    ("back_middle", "back_end"),
    ("back_end", "tail_base"),
    ("tail_base", "tail_end"),
    ("front_left_thai", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),
    ("front_right_thai", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),
    ("back_left_thai", "back_left_knee"),
    ("back_left_knee", "back_left_paw"),
    ("back_right_thai", "back_right_knee"),
    ("back_right_knee", "back_right_paw"),
)


@dataclass(frozen=True)
class Observation39Paths:
    manifest: Path
    trajectory: Path
    report: Path
    preview: Path

    def __iter__(self) -> Iterator[Path]:
        return iter((self.manifest, self.trajectory, self.report, self.preview))


def _paths(root: Path) -> Observation39Paths:
    observation = root / "observation"
    return Observation39Paths(
        manifest=root / "input_manifest.json",
        trajectory=observation / "keypoint_trajectory_2d_39.json",
        report=observation / "keypoint_39_quality_report.json",
        preview=observation / "keypoint_39_preview.mp4",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _role_color(role: str) -> tuple[int, int, int]:
    if "left" in role:
        return (80, 220, 80)
    if "right" in role:
        return (240, 150, 40)
    if role.startswith(("front_", "back_")):
        return (80, 200, 240)
    if role.startswith(("nose", "upper", "lower", "mouth", "neck", "throat")):
        return (80, 100, 245)
    return (220, 190, 70)


def render_39point_preview(
    video_path: Path,
    result: Observation39Result,
    output_path: Path,
) -> None:
    video = result.trajectory["video"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video["fps"]),
        (int(video["width"]), int(video["height"])),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("could not initialize 39-point preview")

    written = 0
    try:
        for semantic_frame in result.trajectory["frames"]:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video ended before frame {written}")
            points = semantic_frame["keypoints"]
            for role_a, role_b in _EDGES:
                point_a = points[role_a]
                point_b = points[role_b]
                if not point_a["valid"] or not point_b["valid"]:
                    continue
                cv2.line(
                    frame,
                    (round(point_a["x_px"]), round(point_a["y_px"])),
                    (round(point_b["x_px"]), round(point_b["y_px"])),
                    (185, 185, 185),
                    1,
                    cv2.LINE_AA,
                )
            invalid = 0
            for role in SUPERANIMAL_QUADRUPED_39:
                point = points[role]
                if not point["valid"]:
                    invalid += 1
                    continue
                cv2.circle(
                    frame,
                    (round(point["x_px"]), round(point["y_px"])),
                    3,
                    _role_color(role),
                    -1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"frame {written:03d}  valid {39 - invalid}/39",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written != video["frame_count"] or not output_path.is_file():
        raise RuntimeError("39-point preview is incomplete")


def run_observation_export(
    *,
    video_path: Path,
    predictions_path: Path,
    mesh_path: Path,
    corrected_spine_path: Path,
    output_dir: Path,
    reference_frame: int = 152,
    individual: str = "animal0",
    confidence_threshold: float = 0.3,
    mesh_generation_method: str = "hunyuan3d_from_video_frame",
) -> Observation39Paths:
    sources = {
        "video": Path(video_path).resolve(),
        "predictions": Path(predictions_path).resolve(),
        "mesh": Path(mesh_path).resolve(),
        "corrected_spine": Path(corrected_spine_path).resolve(),
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} source does not exist: {path}")
    output_dir = Path(output_dir).resolve()
    final = _paths(output_dir)
    for path in final:
        if path.exists():
            raise FileExistsError(
                f"output already exists: {path}; use a new output directory"
            )

    hashes = {label: _sha256(path) for label, path in sources.items()}
    video = probe_video(sources["video"])
    dataframe = pd.read_hdf(sources["predictions"])
    result = build_39point_observation(
        dataframe,
        video,
        individual=individual,
        confidence_threshold=confidence_threshold,
        anchor_frame=reference_frame,
    )
    scorer = result.report["scorer"]
    manifest = {
        "schema": "qianji.cat_39point_input_manifest",
        "schema_version": "0.1.0",
        "reference_frame": reference_frame,
        "sources": {
            label: {"path": str(path), "sha256": hashes[label]}
            for label, path in sources.items()
        },
        "video": result.trajectory["video"],
        "predictions": {
            "scorer": scorer,
            "individual": individual,
            "bodyparts": list(SUPERANIMAL_QUADRUPED_39),
            "coordinates": ["x", "y", "likelihood"],
        },
        "mesh_provenance": {
            "generation_method": mesh_generation_method,
            "provenance_basis": "user_provided",
            "verified_conversion_metadata": "FBX converted to GLB with Assimp",
        },
        "scientific_limits": {
            "reconstruction_kind": "body_relative_2_5d_retarget",
            "metric_depth_observed": False,
            "camera_calibrated": False,
            "global_translation_preserved": False,
            "triangle_mesh_deformed": False,
            "mesh_vertices_deformed": False,
            "mesh_reconstructed_in_pipeline": False,
            "dynamics_simulated": False,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}.observation-staging-",
        dir=output_dir.parent,
    ) as temporary:
        staged = _paths(Path(temporary))
        _write_json(staged.manifest, manifest)
        _write_json(staged.trajectory, result.trajectory)
        _write_json(staged.report, result.report)
        render_39point_preview(sources["video"], result, staged.preview)
        for label, path in sources.items():
            if _sha256(path) != hashes[label]:
                raise RuntimeError(f"{label} source changed during export")
        publish_staged_files(
            tuple(staged),
            tuple(final),
            overwrite=False,
            conflict_hint="use a new output directory",
        )
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--corrected-spine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=152)
    parser.add_argument("--individual", default="animal0")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument(
        "--mesh-generation-method",
        default="hunyuan3d_from_video_frame",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = run_observation_export(
        video_path=args.video,
        predictions_path=args.predictions,
        mesh_path=args.mesh,
        corrected_spine_path=args.corrected_spine,
        output_dir=args.output,
        reference_frame=args.reference_frame,
        individual=args.individual,
        confidence_threshold=args.confidence_threshold,
        mesh_generation_method=args.mesh_generation_method,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
