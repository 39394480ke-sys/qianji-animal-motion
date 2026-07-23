"""Suggest reliable video frames for manually anchoring left/right leg identities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from qianji_animal_motion.semantic_cli import probe_video
from qianji_animal_motion.semantic_mapping import (
    VideoInfo,
    _leg_chains,
    _validate_dataframe,
)


_LEG_COLORS = (
    (80, 220, 80),
    (255, 220, 40),
    (220, 80, 220),
    (255, 120, 60),
)
_LEG_LABELS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class AnchorSuggestionPaths:
    contact_sheet: Path
    manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchor_leg_arrays(
    df: pd.DataFrame,
    *,
    individual: str,
) -> tuple[str, tuple[np.ndarray, ...]]:
    scorer = _validate_dataframe(df, individual)
    front_left, front_right = _leg_chains(df, scorer, individual, "front")
    rear_left, rear_right = _leg_chains(df, scorer, individual, "back")
    return scorer, (front_left, front_right, rear_left, rear_right)


def _torso_scale(legs: tuple[np.ndarray, ...], video: VideoInfo) -> float:
    front_center = (legs[0][:, 0, :2] + legs[1][:, 0, :2]) / 2.0
    rear_center = (legs[2][:, 0, :2] + legs[3][:, 0, :2]) / 2.0
    lengths = np.linalg.norm(front_center - rear_center, axis=1)
    usable = lengths[np.isfinite(lengths) & (lengths > 1.0)]
    if len(usable):
        return float(np.median(usable))
    return math.hypot(video.width, video.height) / 2.0


def _neighbor_motion(
    legs: tuple[np.ndarray, ...],
    frame_idx: int,
    torso_scale: float,
) -> float:
    motions: list[float] = []
    for neighbor in (frame_idx - 1, frame_idx + 1):
        if not 0 <= neighbor < len(legs[0]):
            continue
        for leg in legs:
            current = leg[frame_idx, 2, :2]
            adjacent = leg[neighbor, 2, :2]
            if np.isfinite(current).all() and np.isfinite(adjacent).all():
                motions.append(float(np.linalg.norm(current - adjacent)))
    if not motions:
        return 1.0
    return float(np.median(motions)) / max(torso_scale, 1.0)


def rank_anchor_candidates(
    df: pd.DataFrame,
    video: VideoInfo,
    *,
    individual: str = "animal0",
    confidence_threshold: float = 0.8,
    count: int = 12,
    min_spacing_frames: int | None = None,
) -> list[dict]:
    """Rank clear, separated leg observations and keep temporally diverse frames."""
    if len(df) != video.frame_count:
        raise ValueError("prediction frame count does not match the video")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")
    if count < 1:
        raise ValueError("candidate count must be positive")

    _, legs = _anchor_leg_arrays(df, individual=individual)
    torso_scale = _torso_scale(legs, video)
    minimum_separation = max(5.0, 0.05 * torso_scale)
    eligible: list[dict] = []
    for frame_idx in range(video.frame_count):
        observations = np.stack([leg[frame_idx] for leg in legs])
        if not np.isfinite(observations).all():
            continue
        minimum_confidence = float(np.min(observations[:, :, 2]))
        if minimum_confidence < confidence_threshold:
            continue
        coordinates = observations[:, :, :2]
        if not (
            (coordinates[:, :, 0] >= 0).all()
            and (coordinates[:, :, 0] < video.width).all()
            and (coordinates[:, :, 1] >= 0).all()
            and (coordinates[:, :, 1] < video.height).all()
        ):
            continue

        front_separation = float(
            np.linalg.norm(observations[0, 2, :2] - observations[1, 2, :2])
        )
        rear_separation = float(
            np.linalg.norm(observations[2, 2, :2] - observations[3, 2, :2])
        )
        if min(front_separation, rear_separation) < minimum_separation:
            continue
        motion = _neighbor_motion(legs, frame_idx, torso_scale)
        separation_score = min(front_separation, rear_separation) / max(
            torso_scale, 1.0
        )
        score = minimum_confidence + 0.25 * separation_score - 0.10 * motion
        eligible.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / video.fps,
                "score": float(score),
                "minimum_confidence": minimum_confidence,
                "front_paw_separation_px": front_separation,
                "rear_paw_separation_px": rear_separation,
                "neighbor_motion_normalized": motion,
            }
        )

    spacing = (
        max(1, int(round(video.fps * 0.5)))
        if min_spacing_frames is None
        else max(1, int(min_spacing_frames))
    )
    selected: list[dict] = []
    for candidate in sorted(
        eligible,
        key=lambda item: (-item["score"], item["frame_idx"]),
    ):
        if all(
            abs(candidate["frame_idx"] - chosen["frame_idx"]) >= spacing
            for chosen in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                break
    return sorted(selected, key=lambda item: item["frame_idx"])


def _draw_leg_overlay(
    frame: np.ndarray,
    observations: tuple[np.ndarray, ...],
) -> None:
    for label, color, leg in zip(
        _LEG_LABELS,
        _LEG_COLORS,
        observations,
        strict=True,
    ):
        positions = [(round(point[0]), round(point[1])) for point in leg]
        cv2.line(frame, positions[0], positions[1], color, 2, cv2.LINE_AA)
        cv2.line(frame, positions[1], positions[2], color, 2, cv2.LINE_AA)
        for position in positions:
            cv2.circle(frame, position, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (positions[2][0] + 7, positions[2][1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (positions[2][0] + 7, positions[2][1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            1,
            cv2.LINE_AA,
        )


def render_contact_sheet(
    video_path: Path,
    candidates: list[dict],
    legs: tuple[np.ndarray, ...],
    output_path: Path,
) -> None:
    """Render candidate video frames with raw DLC limb chains and labels."""
    if not candidates:
        raise ValueError("no anchor candidates are available")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    tile_width = 420
    header_height = 34
    tiles: list[np.ndarray] = []
    try:
        for candidate in candidates:
            frame_idx = int(candidate["frame_idx"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read candidate frame {frame_idx}")
            _draw_leg_overlay(
                frame,
                tuple(leg[frame_idx] for leg in legs),
            )
            scale = tile_width / frame.shape[1]
            tile_height = max(1, round(frame.shape[0] * scale))
            resized = cv2.resize(
                frame,
                (tile_width, tile_height),
                interpolation=cv2.INTER_AREA,
            )
            tile = np.full(
                (tile_height + header_height, tile_width, 3),
                245,
                dtype=np.uint8,
            )
            tile[header_height:] = resized
            title = (
                f"frame {frame_idx}  "
                f"time {candidate['timestamp_s']:.3f}s  "
                f"score {candidate['score']:.3f}"
            )
            cv2.putText(
                tile,
                title,
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
    finally:
        capture.release()

    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    tile_height = max(tile.shape[0] for tile in tiles)
    legend_height = 44
    sheet = np.full(
        (legend_height + rows * tile_height, columns * tile_width, 3),
        255,
        dtype=np.uint8,
    )
    x = 10
    for label, color in zip(_LEG_LABELS, _LEG_COLORS, strict=True):
        cv2.circle(sheet, (x + 7, 22), 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            sheet,
            label,
            (x + 20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        x += 82
    cv2.putText(
        sheet,
        "Confirm animal-left/right, not viewer-left/right",
        (x + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y0 = legend_height + row * tile_height
        x0 = column * tile_width
        sheet[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write contact sheet: {output_path}")


def suggest_anchor_candidates(
    *,
    video_path: Path,
    predictions_path: Path,
    output_dir: Path,
    individual: str = "animal0",
    confidence_threshold: float = 0.8,
    count: int = 12,
    min_spacing_frames: int | None = None,
) -> AnchorSuggestionPaths:
    """Generate a review sheet and a source-bound candidate manifest."""
    video_path = Path(video_path).expanduser().resolve()
    predictions_path = Path(predictions_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"DeepLabCut predictions do not exist: {predictions_path}"
        )
    paths = AnchorSuggestionPaths(
        contact_sheet=output_dir / "identity_anchor_candidates.jpg",
        manifest=output_dir / "identity_anchor_candidates.json",
    )
    existing = [path for path in (paths.contact_sheet, paths.manifest) if path.exists()]
    if existing:
        raise FileExistsError(
            f"anchor review output already exists: {existing[0]}; use a new directory"
        )

    video = probe_video(video_path)
    predictions = pd.read_hdf(predictions_path)
    scorer, legs = _anchor_leg_arrays(predictions, individual=individual)
    candidates = rank_anchor_candidates(
        predictions,
        video,
        individual=individual,
        confidence_threshold=confidence_threshold,
        count=count,
        min_spacing_frames=min_spacing_frames,
    )
    if not candidates:
        raise ValueError(
            "no reliable anchor candidates found; lower the confidence threshold "
            "or inspect the DLC predictions"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(video_path, candidates, legs, paths.contact_sheet)
    spacing = (
        max(1, int(round(video.fps * 0.5)))
        if min_spacing_frames is None
        else max(1, int(min_spacing_frames))
    )
    manifest = {
        "schema": "qianji.identity_anchor_candidates",
        "schema_version": "1.0.0",
        "video": {
            "path": str(video_path),
            "sha256": _sha256(video_path),
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frame_count": video.frame_count,
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
            "scorer": scorer,
            "individual": individual,
        },
        "settings": {
            "confidence_threshold": confidence_threshold,
            "requested_count": count,
            "min_spacing_frames": spacing,
        },
        "candidates": candidates,
    }
    paths.manifest.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suggest clear frames for manually anchoring leg identities."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--individual", default="animal0")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--count", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = suggest_anchor_candidates(
        video_path=args.video,
        predictions_path=args.predictions,
        output_dir=args.output,
        individual=args.individual,
        confidence_threshold=args.confidence_threshold,
        count=args.count,
    )
    print(f"contact sheet: {paths.contact_sheet}")
    print(f"manifest: {paths.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
