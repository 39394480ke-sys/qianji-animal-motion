import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import cv2

from qianji_animal_motion.semantic_mapping import (
    REQUIRED_BODYPARTS,
    SEMANTIC_KEYPOINTS,
    VideoInfo,
    _mark_jump_outliers,
    build_semantic_mapping,
    resolve_leg_identities,
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


def _jump_frames(
    x_positions: list[float],
    *,
    invalid_frame: int | None = None,
) -> list[dict]:
    frames = []
    for frame_idx, x_position in enumerate(x_positions):
        keypoints = {}
        for name in SEMANTIC_KEYPOINTS:
            valid = frame_idx != invalid_frame
            keypoints[name] = {
                "x_px": x_position if valid else None,
                "y_px": 0.0 if valid else None,
                "valid": valid,
                "flags": ["low_confidence"] if not valid else [],
                "_candidate_xy": (x_position, 0.0),
            }
        frames.append({"frame_idx": frame_idx, "keypoints": keypoints})
    return frames


def _write_anchor_manifest(
    path: Path,
    *,
    video: Path,
    predictions: Path,
    frame_idx: int = 0,
) -> None:
    payload = {
        "schema": "qianji.identity_anchor_candidates",
        "schema_version": "1.0.0",
        "video": {
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "width": 120,
            "height": 100,
            "fps": 20.0,
            "frame_count": 4,
        },
        "predictions": {
            "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "individual": "animal0",
        },
        "candidates": [
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / 20.0,
                "score": 1.0,
                "minimum_confidence": 0.9,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _leg_at(x: float, confidence: float = 0.9) -> np.ndarray:
    return np.asarray(
        [
            [x, 20.0, confidence],
            [x, 50.0, confidence],
            [x, 80.0, confidence],
        ],
        dtype=float,
    )


def test_anchor_tracks_identities_both_directions_across_full_occlusion() -> None:
    physical_left = [_leg_at(float(index)) for index in range(7)]
    physical_right = [_leg_at(float(index + 20)) for index in range(7)]
    hidden_left = _leg_at(0.0, confidence=0.0)
    hidden_right = _leg_at(0.0, confidence=0.0)
    left = np.stack(
        [
            physical_right[0],
            physical_right[1],
            hidden_left,
            physical_left[3],
            physical_left[4],
            hidden_left,
            physical_right[6],
        ]
    )
    right = np.stack(
        [
            physical_left[0],
            physical_left[1],
            hidden_right,
            physical_right[3],
            physical_right[4],
            hidden_right,
            physical_left[6],
        ]
    )

    identity = resolve_leg_identities(
        left,
        right,
        scale=20.0,
        anchor_frame=3,
        anchor_state=0,
    )

    assert identity.states.tolist() == [1, 1, 0, 0, 0, 0, 1]
    assert identity.ambiguous.tolist() == [False, False, True, False, False, True, False]


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


def test_front_and_rear_anchor_assignments_are_independent() -> None:
    result = build_semantic_mapping(
        _prediction_dataframe(),
        _video_info(),
        anchor_frame=0,
        front_anchor_state=0,
        rear_anchor_state=1,
    )

    points = result.trajectory["frames"][0]["keypoints"]
    assert points["front_left_foot"]["x_px"] == pytest.approx(70.0)
    assert points["rear_left_foot"]["x_px"] == pytest.approx(40.0)
    assert points["rear_left_foot"]["identity_corrected"] is True


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


def test_observed_ambiguous_crossing_does_not_flip_later_identity() -> None:
    def leg_sequence(x_positions: list[float]) -> np.ndarray:
        sequence = np.zeros((len(x_positions), 3, 3), dtype=float)
        sequence[:, :, 2] = 1.0
        for frame, x_position in enumerate(x_positions):
            sequence[frame, :, 0] = x_position
        return sequence

    left = leg_sequence([0.0, 4.9, 5.25, 5.5])
    right = leg_sequence([10.0, 5.1, 4.75, 4.5])

    resolution = resolve_leg_identities(
        left,
        right,
        scale=10.0,
        anchor_frame=0,
        anchor_state=0,
    )

    assert resolution.ambiguous.tolist() == [False, True, False, False]
    assert resolution.states.tolist() == [0, 0, 0, 0]


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


def test_isolated_jump_invalidates_only_center_frame() -> None:
    frames = _jump_frames([0.0, 1.0, 100.0, 3.0, 4.0])

    _mark_jump_outliers(frames, torso_scale=10.0)

    points = [frame["keypoints"]["spine_front"] for frame in frames]
    assert [point["valid"] for point in points] == [
        True,
        True,
        False,
        True,
        True,
    ]
    assert points[2]["flags"] == ["jump_outlier"]
    assert points[3]["flags"] == []


def test_three_frame_isolated_jump_is_detected() -> None:
    frames = _jump_frames([0.0, 100.0, 2.0])

    _mark_jump_outliers(frames, torso_scale=10.0)

    points = [frame["keypoints"]["spine_front"] for frame in frames]
    assert [point["valid"] for point in points] == [True, False, True]
    assert points[1]["flags"] == ["jump_outlier"]


def test_sustained_jump_keeps_single_transition_alert() -> None:
    frames = _jump_frames([0.0, 1.0, 50.0, 51.0, 52.0])

    _mark_jump_outliers(frames, torso_scale=10.0)

    points = [frame["keypoints"]["spine_front"] for frame in frames]
    assert [point["valid"] for point in points] == [
        True,
        True,
        False,
        True,
        True,
    ]
    assert points[2]["flags"] == ["jump_outlier"]
    assert points[3]["flags"] == []


def test_invalid_points_are_excluded_from_jump_detection() -> None:
    frames = _jump_frames(
        [0.0, 1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        invalid_frame=3,
    )

    _mark_jump_outliers(frames, torso_scale=10.0)

    point = frames[3]["keypoints"]["spine_front"]
    assert point["valid"] is False
    assert point["flags"] == ["low_confidence"]


def test_non_finite_prediction_values_are_json_safe() -> None:
    predictions = _prediction_dataframe()
    predictions.loc[
        2,
        ("test_model", "animal0", "front_left_paw", "x"),
    ] = np.inf
    predictions.loc[
        2,
        ("test_model", "animal0", "front_left_paw", "y"),
    ] = -np.inf
    predictions.loc[
        2,
        ("test_model", "animal0", "front_left_paw", "likelihood"),
    ] = np.nan

    result = build_semantic_mapping(predictions, _video_info())

    point = result.trajectory["frames"][2]["keypoints"]["front_left_foot"]
    assert point["x_px"] is None
    assert point["y_px"] is None
    assert point["confidence"] is None
    assert point["valid"] is False
    assert "non_finite" in point["flags"]
    assert result.report["keypoints"]["front_left_foot"][
        "mean_confidence"
    ] == pytest.approx(0.9)
    json.dumps(result.trajectory, allow_nan=False)
    json.dumps(result.report, allow_nan=False)


def test_all_non_finite_confidences_report_null_mean() -> None:
    predictions = _prediction_dataframe()
    predictions.loc[
        :,
        ("test_model", "animal0", "front_left_paw", "likelihood"),
    ] = np.nan

    result = build_semantic_mapping(predictions, _video_info())

    assert all(
        frame["keypoints"]["front_left_foot"]["confidence"] is None
        for frame in result.trajectory["frames"]
    )
    assert (
        result.report["keypoints"]["front_left_foot"]["mean_confidence"] is None
    )


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
    predictions.loc[
        2,
        ("test_model", "animal0", "front_left_paw", "likelihood"),
    ] = np.nan
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
    assert "NaN" not in trajectory_path.read_text(encoding="utf-8")
    assert "Infinity" not in trajectory_path.read_text(encoding="utf-8")
    assert "NaN" not in report_path.read_text(encoding="utf-8")
    assert "Infinity" not in report_path.read_text(encoding="utf-8")
    assert payload["schema"] == "qianji.keypoint_trajectory_2d"
    assert payload["video"]["frame_count"] == 4
    assert payload["frames"][2]["keypoints"]["front_left_foot"]["confidence"] is None
    assert report["schema"] == "qianji.keypoint_mapping_report"
    assert source.read_bytes() == before


def test_json_outputs_are_serialized_before_either_file_is_written(
    tmp_path: Path,
) -> None:
    source = tmp_path / "predictions.h5"
    source.write_bytes(b"h5 fixture")
    result = build_semantic_mapping(_prediction_dataframe(), _video_info())
    result.report["unexpected_non_finite"] = np.nan
    output_dir = tmp_path / "semantic"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_outputs(
            result,
            output_dir=output_dir,
            source_h5=source,
            source_video=tmp_path / "video.mp4",
        )

    assert not (output_dir / "keypoint_trajectory_2d.json").exists()
    assert not (output_dir / "mapping_report.json").exists()


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
    anchor_manifest = tmp_path / "identity_anchor_candidates.json"
    _write_anchor_manifest(
        anchor_manifest,
        video=video_path,
        predictions=predictions_path,
    )

    outputs = run_mapping(
        video_path,
        predictions_path,
        output_dir,
        anchor_manifest=anchor_manifest,
        anchor_frame=0,
        front_anchor="keep",
        rear_anchor="keep",
    )

    assert outputs.trajectory.name == "keypoint_trajectory_2d.json"
    assert outputs.report.name == "mapping_report.json"
    assert outputs.preview.name == "six_keypoints_preview.mp4"
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    capture = cv2.VideoCapture(str(outputs.preview))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0)
    capture.release()
    assert predictions_path.read_bytes() == original_predictions
    trajectory = json.loads(outputs.trajectory.read_text(encoding="utf-8"))
    assert trajectory["identity_anchor"]["frame_idx"] == 0
    assert trajectory["identity_anchor"]["front_assignment"] == "keep"
    assert trajectory["identity_anchor"]["rear_assignment"] == "keep"


def test_run_mapping_requires_a_manual_anchor(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    predictions = tmp_path / "predictions.h5"
    video.write_bytes(b"video")
    predictions.write_bytes(b"predictions")

    with pytest.raises(ValueError, match="anchor manifest"):
        run_mapping(video, predictions, tmp_path / "outputs")


def test_run_mapping_rejects_anchor_manifest_for_another_video(
    tmp_path: Path,
) -> None:
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
    manifest = tmp_path / "identity_anchor_candidates.json"
    _write_anchor_manifest(
        manifest,
        video=video_path,
        predictions=predictions_path,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["video"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="video hash"):
        run_mapping(
            video_path,
            predictions_path,
            tmp_path / "outputs",
            anchor_manifest=manifest,
            anchor_frame=0,
            front_anchor="keep",
            rear_anchor="keep",
        )


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
