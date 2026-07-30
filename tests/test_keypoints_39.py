from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from qianji_animal_motion.keypoints_39 import (
    SUPERANIMAL_QUADRUPED_39,
    build_39point_observation,
)
from qianji_animal_motion.keypoints_39_cli import run_observation_export
from qianji_animal_motion.semantic_mapping import VideoInfo


EXPECTED_ROLES = (
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "tail_base",
    "tail_end",
    "front_left_thai",
    "front_left_knee",
    "front_left_paw",
    "front_right_thai",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_thai",
    "back_right_thai",
    "back_left_knee",
    "back_right_knee",
    "back_right_paw",
    "belly_bottom",
    "body_middle_right",
    "body_middle_left",
)


def _video(frame_count: int) -> VideoInfo:
    return VideoInfo(width=100, height=80, fps=30.0, frame_count=frame_count)


def _values(frame_count: int) -> dict[str, np.ndarray]:
    values = {}
    for index, role in enumerate(EXPECTED_ROLES):
        x = np.full(frame_count, 10.0 + index, dtype=float)
        y = np.full(frame_count, 20.0 + (index % 5), dtype=float)
        confidence = np.full(frame_count, 0.9, dtype=float)
        values[role] = np.column_stack([x, y, confidence])
    values["back_end"][:, 0] = 20.0
    values["back_base"][:, 0] = 60.0
    values["back_end"][:, 1] = 30.0
    values["back_base"][:, 1] = 30.0
    return values


def _dataframe(values: dict[str, np.ndarray]) -> pd.DataFrame:
    roles = tuple(values)
    frame_count = len(values[roles[0]])
    columns = pd.MultiIndex.from_product(
        [
            ["scorer"],
            ["animal0"],
            roles,
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    rows = np.empty((frame_count, len(columns)), dtype=float)
    for role_index, role in enumerate(roles):
        rows[:, role_index * 3 : role_index * 3 + 3] = values[role]
    return pd.DataFrame(rows, columns=columns)


def test_preserves_all_39_roles_in_every_frame() -> None:
    dataframe = _dataframe(_values(2))

    result = build_39point_observation(
        dataframe,
        _video(2),
        anchor_frame=0,
    )

    assert SUPERANIMAL_QUADRUPED_39 == EXPECTED_ROLES
    assert result.trajectory["schema"] == "qianji.keypoint_trajectory_2d_39"
    assert len(result.trajectory["frames"]) == 2
    for frame in result.trajectory["frames"]:
        assert tuple(frame["keypoints"]) == EXPECTED_ROLES
        assert len(frame["keypoints"]) == 39


def test_quality_policy_keeps_bad_observations_explicit() -> None:
    values = _values(2)
    values["nose"][0] = [10.0, 20.0, 0.29]
    values["upper_jaw"][0] = [101.0, 20.0, 0.9]
    values["lower_jaw"][0] = [10.0, 20.0, 0.9]
    values["lower_jaw"][1] = [90.0, 20.0, 0.9]

    result = build_39point_observation(
        _dataframe(values),
        _video(2),
        confidence_threshold=0.3,
        anchor_frame=0,
    )

    frame0 = result.trajectory["frames"][0]["keypoints"]
    frame1 = result.trajectory["frames"][1]["keypoints"]
    assert frame0["nose"]["valid"] is False
    assert frame0["nose"]["x_px"] is None
    assert frame0["nose"]["raw_x_px"] == 10.0
    assert frame0["nose"]["flags"] == ["low_confidence"]
    assert frame0["upper_jaw"]["valid"] is False
    assert frame0["upper_jaw"]["flags"] == ["out_of_bounds"]
    assert frame1["lower_jaw"]["valid"] is False
    assert frame1["lower_jaw"]["flags"] == ["temporal_jump"]
    assert result.report["roles"]["nose"]["invalid_frames"] == [0]
    assert result.report["roles"]["nose"]["flag_counts"] == {
        "low_confidence": 1
    }


def test_identity_swap_applies_to_complete_front_and_rear_chains() -> None:
    values = _values(2)
    for prefix in ("front", "back"):
        for joint_index, joint in enumerate(("thai", "knee", "paw")):
            left = f"{prefix}_left_{joint}"
            right = f"{prefix}_right_{joint}"
            values[left][0, :2] = [10.0 + joint_index, 50.0]
            values[right][0, :2] = [80.0 + joint_index, 50.0]
            values[left][1, :2] = [81.0 + joint_index, 50.0]
            values[right][1, :2] = [11.0 + joint_index, 50.0]

    result = build_39point_observation(
        _dataframe(values),
        _video(2),
        anchor_frame=0,
    )

    frame = result.trajectory["frames"][1]["keypoints"]
    for prefix in ("front", "back"):
        for joint_index, joint in enumerate(("thai", "knee", "paw")):
            left = frame[f"{prefix}_left_{joint}"]
            right = frame[f"{prefix}_right_{joint}"]
            assert left["raw_x_px"] == 11.0 + joint_index
            assert right["raw_x_px"] == 81.0 + joint_index
            assert left["identity_corrected"] is True
            assert right["identity_corrected"] is True
    assert result.report["identity"]["front"]["swapped_frames"] == [1]
    assert result.report["identity"]["rear"]["swapped_frames"] == [1]


def test_rejects_role_loss_and_malformed_metadata() -> None:
    values = _values(2)
    missing = copy.deepcopy(values)
    del missing["nose"]

    with pytest.raises(ValueError, match="bodyparts"):
        build_39point_observation(
            _dataframe(missing),
            _video(2),
            anchor_frame=0,
        )

    with pytest.raises(ValueError, match="frame_count"):
        build_39point_observation(
            _dataframe(values),
            _video(3),
            anchor_frame=0,
        )


def _write_video(path: Path, frame_count: int = 2) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (100, 80),
    )
    assert writer.isOpened()
    for frame_idx in range(frame_count):
        frame = np.full((80, 100, 3), 20 + 10 * frame_idx, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_observation_export_publishes_hashed_manifest_and_preview(
    tmp_path: Path,
) -> None:
    video = tmp_path / "cat.mp4"
    predictions = tmp_path / "predictions.h5"
    mesh = tmp_path / "cat.glb"
    corrected_spine = tmp_path / "corrected.json"
    _write_video(video)
    _dataframe(_values(2)).to_hdf(predictions, key="df")
    mesh.write_bytes(b"mesh")
    corrected_spine.write_text(
        json.dumps({"identity_anchor": {"frame_idx": 1}}),
        encoding="utf-8",
    )

    outputs = run_observation_export(
        video_path=video,
        predictions_path=predictions,
        mesh_path=mesh,
        corrected_spine_path=corrected_spine,
        output_dir=tmp_path / "case",
        reference_frame=1,
        mesh_generation_method="hunyuan3d_from_video_frame",
    )

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["reference_frame"] == 1
    assert manifest["mesh_provenance"]["generation_method"] == (
        "hunyuan3d_from_video_frame"
    )
    for label, path in {
        "video": video,
        "predictions": predictions,
        "mesh": mesh,
        "corrected_spine": corrected_spine,
    }.items():
        assert manifest["sources"][label] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    capture = cv2.VideoCapture(str(outputs.preview))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 100
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 80
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert len(decoded) == 2
    assert int(decoded[0].max()) > 100

    with pytest.raises(FileExistsError, match="output already exists"):
        run_observation_export(
            video_path=video,
            predictions_path=predictions,
            mesh_path=mesh,
            corrected_spine_path=corrected_spine,
            output_dir=tmp_path / "case",
            reference_frame=1,
            mesh_generation_method="hunyuan3d_from_video_frame",
        )
