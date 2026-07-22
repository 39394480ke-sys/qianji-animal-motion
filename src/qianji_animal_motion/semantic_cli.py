"""CLI orchestration and preview rendering for six-keypoint trajectories."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import pandas as pd

from qianji_animal_motion.semantic_mapping import (
    MappingResult,
    VideoInfo,
    build_semantic_mapping,
    write_json_outputs,
)


@dataclass(frozen=True)
class OutputPaths:
    trajectory: Path
    report: Path
    preview: Path

    def __iter__(self) -> Iterator[Path]:
        return iter((self.trajectory, self.report, self.preview))


_COLORS = {
    "spine_front": (0, 215, 255),
    "spine_rear": (0, 140, 255),
    "front_left_foot": (80, 220, 80),
    "front_right_foot": (255, 220, 40),
    "rear_left_foot": (220, 80, 220),
    "rear_right_foot": (255, 120, 60),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def render_preview(
    video_path: Path,
    result: MappingResult,
    output_path: Path,
) -> None:
    video = result.trajectory["video"]
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
        raise RuntimeError("could not initialize preview video reader or writer")

    written = 0
    try:
        for semantic_frame in result.trajectory["frames"]:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video ended before frame {written}")
            points = semantic_frame["keypoints"]
            front = points["spine_front"]
            rear = points["spine_rear"]
            if front["valid"] and rear["valid"]:
                cv2.line(
                    frame,
                    (round(rear["x_px"]), round(rear["y_px"])),
                    (round(front["x_px"]), round(front["y_px"])),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            invalid_names = []
            for name, point in points.items():
                if not point["valid"]:
                    invalid_names.append(name)
                    continue
                position = (round(point["x_px"]), round(point["y_px"]))
                color = _COLORS[name]
                cv2.circle(frame, position, 7, color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    name,
                    (position[0] + 9, position[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    name,
                    (position[0] + 9, position[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"frame {semantic_frame['frame_idx']}",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if invalid_names:
                cv2.putText(
                    frame,
                    "invalid: " + ", ".join(invalid_names),
                    (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written != int(video["frame_count"]):
        raise RuntimeError(
            f"preview frame count mismatch: wrote {written}, expected {video['frame_count']}"
        )


def run_mapping(
    video_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    individual: str = "animal0",
    confidence_threshold: float = 0.5,
    overwrite: bool = False,
) -> OutputPaths:
    video_path = Path(video_path).expanduser().resolve()
    predictions_path = Path(predictions_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    outputs = OutputPaths(
        trajectory=output_dir / "keypoint_trajectory_2d.json",
        report=output_dir / "mapping_report.json",
        preview=output_dir / "six_keypoints_preview.mp4",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output already exists: {existing[0]}; use --overwrite to replace it"
        )
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"DeepLabCut predictions do not exist: {predictions_path}"
        )
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")

    before_hash = _file_sha256(predictions_path)
    video = probe_video(video_path)
    predictions = pd.read_hdf(predictions_path)
    result = build_semantic_mapping(
        predictions,
        video,
        individual=individual,
        confidence_threshold=confidence_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path, report_path = write_json_outputs(
        result,
        output_dir=output_dir,
        source_h5=predictions_path,
        source_video=video_path,
    )
    render_preview(video_path, result, outputs.preview)
    after_hash = _file_sha256(predictions_path)
    if before_hash != after_hash:
        raise RuntimeError("source DeepLabCut predictions changed during mapping")
    return OutputPaths(
        trajectory=trajectory_path,
        report=report_path,
        preview=outputs.preview,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map DeepLabCut quadruped predictions to six QianJi keypoints."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--individual", default="animal0")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_mapping(
        args.video,
        args.predictions,
        args.output,
        individual=args.individual,
        confidence_threshold=args.confidence_threshold,
        overwrite=args.overwrite,
    )
    print(f"trajectory: {outputs.trajectory}")
    print(f"report: {outputs.report}")
    print(f"preview: {outputs.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
