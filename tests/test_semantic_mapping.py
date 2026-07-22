import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import cv2

from qianji_animal_motion.semantic_mapping import (
    REQUIRED_BODYPARTS,
    VideoInfo,
    build_semantic_mapping,
    write_json_outputs,
)
from qianji_animal_motion.semantic_cli import run_mapping


def _prediction_dataframe(frame_count: int = 4) -> pd.DataFrame:
    scorer = "test_model"
    individual = "animal0"
    columns = pd.MultiIndex.from_product(
        [[scorer], [individual], REQUIRED_BODYPARTS, ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    data = np.zeros((frame_count, len(columns)), dtype=float)
    frame = pd.DataFrame(data, columns=columns)

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
            frame.loc[index, (scorer, individual, bodypart, "x")] = x + index
            frame.loc[index, (scorer, individual, bodypart, "y")] = y
            frame.loc[index, (scorer, individual, bodypart, "likelihood")] = 0.9
    return frame


def _swap_pair(frame: pd.DataFrame, index: int, prefix: str) -> None:
    scorer = "test_model"
    individual = "animal0"
    for joint in ("thai", "knee", "paw"):
        left = f"{prefix}_left_{joint}"
        right = f"{prefix}_right_{joint}"
        for coord in ("x", "y", "likelihood"):
            left_key = (scorer, individual, left, coord)
            right_key = (scorer, individual, right, coord)
            frame.loc[index, left_key], frame.loc[index, right_key] = (
                frame.loc[index, right_key],
                frame.loc[index, left_key],
            )


def _video_info(frame_count: int = 4) -> VideoInfo:
    return VideoInfo(width=120, height=100, fps=20.0, frame_count=frame_count)


def test_maps_dorsal_landmarks_directly_to_spine_points() -> None:
    result = build_semantic_mapping(_prediction_dataframe(), _video_info())

    points = result.trajectory["frames"][0]["keypoints"]
    assert points["spine_front"]["x_px"] == pytest.approx(85.0)
    assert points["spine_front"]["y_px"] == pytest.approx(20.0)
    assert points["spine_front"]["confidence"] == pytest.approx(0.9)
    assert points["spine_front"]["source"] == "back_base"
    assert points["spine_front"]["fallback_used"] is False
    assert points["spine_rear"]["x_px"] == pytest.approx(25.0)
    assert points["spine_rear"]["source"] == "back_end"


def test_uses_validated_same_frame_fallback_for_low_confidence_spine() -> None:
    predictions = _prediction_dataframe()
    predictions.loc[
        2,
        ("test_model", "animal0", "back_base", "likelihood"),
    ] = 0.2

    result = build_semantic_mapping(predictions, _video_info())

    point = result.trajectory["frames"][2]["keypoints"]["spine_front"]
    assert point["x_px"] == pytest.approx(87.0)
    assert point["y_px"] == pytest.approx(20.0)
    assert point["source"] == "neck_end"
    assert point["fallback_used"] is True
    assert point["valid"] is True
    assert result.report["keypoints"]["spine_front"]["fallback_used_frames"] == [2]


def test_nulls_spine_when_primary_and_fallback_are_unavailable() -> None:
    predictions = _prediction_dataframe()
    for bodypart in ("back_base", "neck_end", "back_middle"):
        predictions.loc[
            2,
            ("test_model", "animal0", bodypart, "likelihood"),
        ] = 0.2

    result = build_semantic_mapping(predictions, _video_info())

    point = result.trajectory["frames"][2]["keypoints"]["spine_front"]
    assert point["x_px"] is None
    assert point["y_px"] is None
    assert point["valid"] is False
    assert point["fallback_used"] is False
    assert "low_confidence" in point["flags"]
    assert "fallback_unavailable" in point["flags"]


def test_repairs_front_spine_from_validated_same_frame_back_structure() -> None:
    predictions = _prediction_dataframe()
    for bodypart in ("back_base", "neck_end"):
        predictions.loc[
            2,
            ("test_model", "animal0", bodypart, "likelihood"),
        ] = 0.2

    result = build_semantic_mapping(predictions, _video_info())

    point = result.trajectory["frames"][2]["keypoints"]["spine_front"]
    assert point["x_px"] == pytest.approx(87.0)
    assert point["y_px"] == pytest.approx(20.0)
    assert point["source"] == ["back_end", "back_middle"]
    assert point["fallback_used"] is True
    assert point["valid"] is True


def test_rejects_spine_geometry_that_does_not_follow_back_polyline() -> None:
    predictions = _prediction_dataframe()
    predictions.loc[2, ("test_model", "animal0", "back_middle", "y")] = 90.0

    result = build_semantic_mapping(predictions, _video_info())

    front = result.trajectory["frames"][2]["keypoints"]["spine_front"]
    rear = result.trajectory["frames"][2]["keypoints"]["spine_rear"]
    assert front["valid"] is False
    assert rear["valid"] is False
    assert "back_polyline_inconsistent" in front["flags"]
    assert "back_polyline_inconsistent" in rear["flags"]


def test_corrects_a_single_frame_left_right_swap() -> None:
    predictions = _prediction_dataframe()
    _swap_pair(predictions, index=2, prefix="back")

    result = build_semantic_mapping(predictions, _video_info())

    frame = result.trajectory["frames"][2]
    left = frame["keypoints"]["rear_left_foot"]
    right = frame["keypoints"]["rear_right_foot"]
    assert left["x_px"] == pytest.approx(12.0)
    assert right["x_px"] == pytest.approx(42.0)
    assert left["identity_corrected"] is True
    assert right["identity_corrected"] is True
    assert result.report["identity_corrections"]["rear"] == [2]


def test_ambiguous_identity_is_null_and_flagged() -> None:
    predictions = _prediction_dataframe()
    scorer = "test_model"
    individual = "animal0"
    for joint in ("thai", "knee", "paw"):
        left = f"back_left_{joint}"
        right = f"back_right_{joint}"
        for coord in ("x", "y"):
            value = predictions.loc[0, (scorer, individual, left, coord)]
            predictions.loc[1, (scorer, individual, left, coord)] = value
            predictions.loc[1, (scorer, individual, right, coord)] = value

    result = build_semantic_mapping(predictions, _video_info())

    point = result.trajectory["frames"][1]["keypoints"]["rear_left_foot"]
    assert point["x_px"] is None
    assert point["y_px"] is None
    assert point["valid"] is False
    assert "identity_ambiguous" in point["flags"]


def test_low_confidence_point_is_null_without_interpolation() -> None:
    predictions = _prediction_dataframe()
    predictions.loc[
        2,
        ("test_model", "animal0", "front_left_paw", "likelihood"),
    ] = 0.2

    result = build_semantic_mapping(predictions, _video_info())

    points = result.trajectory["frames"]
    invalid = points[2]["keypoints"]["front_left_foot"]
    assert invalid["x_px"] is None
    assert invalid["y_px"] is None
    assert invalid["confidence"] == pytest.approx(0.2)
    assert invalid["valid"] is False
    assert "low_confidence" in invalid["flags"]
    assert points[1]["keypoints"]["front_left_foot"]["x_px"] is not None
    assert points[3]["keypoints"]["front_left_foot"]["x_px"] is not None


def test_rejects_paw_that_is_not_distal_to_its_knee() -> None:
    predictions = _prediction_dataframe()
    for coord, value in (("x", 82.0), ("y", 42.0)):
        predictions.loc[
            2,
            ("test_model", "animal0", "front_left_paw", coord),
        ] = value

    result = build_semantic_mapping(predictions, _video_info())

    point = result.trajectory["frames"][2]["keypoints"]["front_left_foot"]
    assert point["valid"] is False
    assert point["x_px"] is None
    assert "limb_structure_inconsistent" in point["flags"]


def test_writes_json_outputs_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "predictions.h5"
    predictions = _prediction_dataframe()
    predictions.to_hdf(source, key="df_with_missing", mode="w")
    before = source.read_bytes()
    result = build_semantic_mapping(predictions, _video_info())

    trajectory_path, report_path = write_json_outputs(
        result,
        output_dir=tmp_path / "semantic",
        source_h5=source,
        source_video=tmp_path / "video.mp4",
    )

    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "qianji.keypoint_trajectory_2d"
    assert payload["video"]["frame_count"] == 4
    assert report["schema"] == "qianji.keypoint_mapping_report"
    assert source.read_bytes() == before


def test_run_mapping_creates_json_report_and_preview(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (120, 100),
    )
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((100, 120, 3), 240, dtype=np.uint8))
    writer.release()
    predictions_path = tmp_path / "predictions.h5"
    _prediction_dataframe().to_hdf(
        predictions_path,
        key="df_with_missing",
        mode="w",
    )
    original_predictions = predictions_path.read_bytes()
    output_dir = tmp_path / "outputs"

    outputs = run_mapping(video_path, predictions_path, output_dir)

    assert outputs.trajectory.name == "keypoint_trajectory_2d.json"
    assert outputs.report.name == "mapping_report.json"
    assert outputs.preview.name == "six_keypoints_preview.mp4"
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    capture = cv2.VideoCapture(str(outputs.preview))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0)
    capture.release()
    assert predictions_path.read_bytes() == original_predictions


def test_run_mapping_refuses_to_overwrite_existing_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "mapping_report.json").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_mapping(
            tmp_path / "missing-video.mp4",
            tmp_path / "missing-predictions.h5",
            output_dir,
        )
