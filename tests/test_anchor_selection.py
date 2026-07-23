import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from qianji_animal_motion.anchor_selection import (
    rank_anchor_candidates,
    suggest_anchor_candidates,
)
from qianji_animal_motion.semantic_mapping import REQUIRED_BODYPARTS, VideoInfo


def _predictions(frame_count: int = 8) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [
            ["test_model"],
            ["animal0"],
            REQUIRED_BODYPARTS,
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    frame = pd.DataFrame(
        np.zeros((frame_count, len(columns)), dtype=float),
        columns=columns,
    )
    defaults = {
        "back_base": (85.0, 20.0),
        "back_middle": (55.0, 20.0),
        "back_end": (25.0, 20.0),
        "neck_end": (85.0, 20.0),
        "tail_base": (25.0, 20.0),
        "front_left_thai": (80.0, 40.0),
        "front_left_knee": (75.0, 65.0),
        "front_left_paw": (70.0, 90.0),
        "front_right_thai": (90.0, 40.0),
        "front_right_knee": (95.0, 65.0),
        "front_right_paw": (100.0, 90.0),
        "back_left_thai": (20.0, 40.0),
        "back_left_knee": (15.0, 65.0),
        "back_left_paw": (10.0, 90.0),
        "back_right_thai": (30.0, 40.0),
        "back_right_knee": (35.0, 65.0),
        "back_right_paw": (40.0, 90.0),
    }
    for bodypart, (x, y) in defaults.items():
        for index in range(frame_count):
            frame.loc[
                index,
                ("test_model", "animal0", bodypart, "x"),
            ] = x + index
            frame.loc[
                index,
                ("test_model", "animal0", bodypart, "y"),
            ] = y
            frame.loc[
                index,
                ("test_model", "animal0", bodypart, "likelihood"),
            ] = 0.95
    return frame


def _write_video(path: Path, frame_count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (120, 100),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((100, 120, 3), 220 - index, dtype=np.uint8))
    writer.release()


def test_candidate_ranking_rejects_low_confidence_and_overlapping_legs() -> None:
    predictions = _predictions()
    predictions.loc[
        2,
        ("test_model", "animal0", "front_left_paw", "likelihood"),
    ] = 0.4
    for joint in ("thai", "knee", "paw"):
        for coord in ("x", "y"):
            predictions.loc[
                3,
                ("test_model", "animal0", f"front_right_{joint}", coord),
            ] = predictions.loc[
                3,
                ("test_model", "animal0", f"front_left_{joint}", coord),
            ]

    candidates = rank_anchor_candidates(
        predictions,
        VideoInfo(width=120, height=100, fps=20.0, frame_count=8),
        confidence_threshold=0.8,
        count=3,
        min_spacing_frames=2,
    )

    frames = [candidate["frame_idx"] for candidate in candidates]
    assert len(frames) == 3
    assert 2 not in frames
    assert 3 not in frames
    assert all(abs(left - right) >= 2 for left in frames for right in frames if left != right)


def test_suggestion_package_writes_contact_sheet_and_hashed_manifest(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    predictions = tmp_path / "predictions.h5"
    output = tmp_path / "anchor_review"
    _write_video(video)
    _predictions().to_hdf(predictions, key="df_with_missing", mode="w")

    paths = suggest_anchor_candidates(
        video_path=video,
        predictions_path=predictions,
        output_dir=output,
        camera_side="animal_left_visible",
        count=4,
        min_spacing_frames=2,
    )

    assert paths.contact_sheet.is_file()
    assert paths.contact_sheet.stat().st_size > 0
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == "qianji.identity_anchor_candidates"
    assert manifest["video"]["frame_count"] == 8
    assert len(manifest["video"]["sha256"]) == 64
    assert len(manifest["predictions"]["sha256"]) == 64
    assert len(manifest["candidates"]) == 4
    assert manifest["settings"]["camera_side"] == "animal_left_visible"


def test_suggestion_package_discards_partial_contact_sheet_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "video.mp4"
    predictions = tmp_path / "predictions.h5"
    output = tmp_path / "anchor_review"
    _write_video(video)
    _predictions().to_hdf(predictions, key="df_with_missing", mode="w")

    def fail_render(
        _video: Path,
        _candidates: list[dict],
        _legs: tuple[np.ndarray, ...],
        output_path: Path,
    ) -> None:
        output_path.write_bytes(b"partial")
        raise RuntimeError("contact sheet failed")

    monkeypatch.setattr(
        "qianji_animal_motion.anchor_selection.render_contact_sheet",
        fail_render,
    )

    with pytest.raises(RuntimeError, match="contact sheet failed"):
        suggest_anchor_candidates(
            video_path=video,
            predictions_path=predictions,
            output_dir=output,
            camera_side="unknown",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".anchor_review.staging-*"))


def test_suggestion_package_rejects_source_changes_during_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "video.mp4"
    predictions = tmp_path / "predictions.h5"
    output = tmp_path / "anchor_review"
    _write_video(video)
    _predictions().to_hdf(predictions, key="df_with_missing", mode="w")

    def mutate_video(
        _video: Path,
        _candidates: list[dict],
        _legs: tuple[np.ndarray, ...],
        output_path: Path,
    ) -> None:
        output_path.write_bytes(b"contact sheet")
        video.write_bytes(b"changed during render")

    monkeypatch.setattr(
        "qianji_animal_motion.anchor_selection.render_contact_sheet",
        mutate_video,
    )

    with pytest.raises(RuntimeError, match="source video changed"):
        suggest_anchor_candidates(
            video_path=video,
            predictions_path=predictions,
            output_dir=output,
            camera_side="unknown",
        )

    assert not output.exists()
