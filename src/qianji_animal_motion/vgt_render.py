"""OpenCV renderer for the real 12-site/30-rod VGT structure motion."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from qianji_animal_motion.vgt_sequence import VgtSequence, validate_render_rate


_BACKGROUND = (250, 250, 248)
_ROD_COLOR = (65, 77, 92)
_SITE_COLOR = (40, 120, 215)
_SITE_OUTLINE = (20, 35, 50)


def _projections(position: np.ndarray) -> tuple[np.ndarray, ...]:
    top = position[:, [0, 1]]
    side = position[:, [0, 2]]
    front = position[:, [1, 2]]
    iso_x = 0.82 * position[:, 0] - 0.58 * position[:, 1]
    iso_y = (
        0.34 * position[:, 0]
        + 0.48 * position[:, 1]
        - 0.82 * position[:, 2]
    )
    isometric = np.stack([iso_x, iso_y], axis=1)
    return top, side, front, isometric


def _bounds(sequence: VgtSequence) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    projected = [_projections(frame) for frame in sequence.positions]
    output = []
    for view_index in range(4):
        values = np.concatenate(
            [frame[view_index] for frame in projected],
            axis=0,
        )
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        span = maximum - minimum
        span[span < 1e-6] = 1.0
        output.append((minimum - 0.08 * span, maximum + 0.08 * span))
    return tuple(output)


def _map_points(
    values: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    width: int,
    height: int,
    *,
    padding: int,
) -> np.ndarray:
    minimum, maximum = bounds
    usable_width = max(1, width - 2 * padding)
    usable_height = max(1, height - 2 * padding)
    span = maximum - minimum
    scale = min(usable_width / span[0], usable_height / span[1])
    occupied = span * scale
    offset = np.asarray(
        [
            (width - occupied[0]) / 2.0,
            (height - occupied[1]) / 2.0,
        ]
    )
    mapped = (values - minimum) * scale + offset
    mapped[:, 1] = height - mapped[:, 1]
    return np.rint(mapped).astype(np.int32)


def _draw_panel(
    position: np.ndarray,
    rods: tuple[tuple[int, int], ...],
    view_index: int,
    bounds: tuple[np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    values = _projections(position)[view_index]
    points = _map_points(
        values,
        bounds,
        width,
        height,
        padding=max(14, min(width, height) // 14),
    )
    line_width = max(1, min(width, height) // 180)
    radius = max(3, min(width, height) // 70)
    for site1, site2 in rods:
        cv2.line(
            panel,
            tuple(points[site1]),
            tuple(points[site2]),
            _ROD_COLOR,
            line_width,
            cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(
            panel,
            tuple(point),
            radius + 1,
            _SITE_OUTLINE,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            panel,
            tuple(point),
            radius,
            _SITE_COLOR,
            -1,
            cv2.LINE_AA,
        )
    return panel


def _rod_indices(sequence: VgtSequence) -> tuple[tuple[int, int], ...]:
    indices = {name: index for index, name in enumerate(sequence.site_names)}
    return tuple((indices[site1], indices[site2]) for site1, site2 in sequence.rods)


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image) or not path.is_file():
        raise RuntimeError(f"failed to write VGT image: {path}")


def render_vgt_motion(
    sequence: VgtSequence,
    robot: dict,
    output_dir: Path,
    *,
    width: int = 1246,
    height: int = 720,
    fps: float = 30.0,
) -> dict:
    """Render static evidence and an exact-rate four-view structure video."""
    if width < 320 or height < 240:
        raise ValueError("render dimensions are too small")
    if len(sequence.rods) != len(robot.get("rod_groups", ())):
        raise ValueError("sequence and robot rod counts differ")
    validate_render_rate(sequence, fps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    three_view_path = output_dir / "vgt_three_view.png"
    isometric_path = output_dir / "vgt_isometric.png"
    video_path = output_dir / "vgt_motion_30fps.mp4"
    for path in (three_view_path, isometric_path, video_path):
        if path.exists():
            raise FileExistsError(f"render output already exists: {path}")

    rods = _rod_indices(sequence)
    bounds = _bounds(sequence)
    first = sequence.positions[0]
    static_panels = [
        _draw_panel(first, rods, index, bounds[index], 400, 400)
        for index in range(4)
    ]
    titles = ("Top XY", "Side XZ", "Front YZ")
    for panel, title in zip(static_panels[:3], titles, strict=True):
        cv2.putText(
            panel,
            title,
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 40, 50),
            1,
            cv2.LINE_AA,
        )
    three_view = np.concatenate(static_panels[:3], axis=1)
    cv2.putText(
        static_panels[3],
        "Isometric",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 40, 50),
        1,
        cv2.LINE_AA,
    )
    _write_image(three_view_path, three_view)
    _write_image(isometric_path, static_panels[3])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VGT video writer: {video_path}")
    left_width = width // 2
    right_width = width - left_width
    top_height = height // 2
    bottom_height = height - top_height
    frame_count = 0
    try:
        for position in sequence.positions:
            panels = (
                _draw_panel(position, rods, 0, bounds[0], left_width, top_height),
                _draw_panel(position, rods, 1, bounds[1], right_width, top_height),
                _draw_panel(
                    position,
                    rods,
                    2,
                    bounds[2],
                    left_width,
                    bottom_height,
                ),
                _draw_panel(
                    position,
                    rods,
                    3,
                    bounds[3],
                    right_width,
                    bottom_height,
                ),
            )
            canvas = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
            canvas[:top_height, :left_width] = panels[0]
            canvas[:top_height, left_width:] = panels[1]
            canvas[top_height:, :left_width] = panels[2]
            canvas[top_height:, left_width:] = panels[3]
            writer.write(canvas)
            frame_count += 1
    finally:
        writer.release()
    if frame_count != sequence.positions.shape[0] or not video_path.is_file():
        raise RuntimeError("VGT video was only partially written")
    capture = cv2.VideoCapture(str(video_path))
    decoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    decoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if (
        decoded_count != frame_count
        or not math.isclose(decoded_fps, fps, abs_tol=0.01)
        or decoded_width != width
        or decoded_height != height
    ):
        raise RuntimeError("written VGT video metadata does not match request")
    return {
        "frame_count": frame_count,
        "site_count": len(sequence.site_names),
        "rod_count": len(sequence.rods),
        "fps": float(fps),
        "width": width,
        "height": height,
        "three_view": str(three_view_path),
        "isometric": str(isometric_path),
        "video": str(video_path),
    }
