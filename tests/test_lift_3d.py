import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from qianji_animal_motion.lift_3d import (
    KEYPOINT_ROLES,
    lift_trajectory,
    neutral_pose_from_rig,
    select_reference_frame,
)
from qianji_animal_motion.lift_cli import run_lift


NEUTRAL = {
    "spine_rear": [-0.5, 0.0, 0.8],
    "spine_front": [0.5, 0.0, 0.8],
    "front_left_foot": [0.5, 0.2, 0.0],
    "front_right_foot": [0.5, -0.2, 0.0],
    "rear_left_foot": [-0.5, 0.2, 0.0],
    "rear_right_foot": [-0.5, -0.2, 0.0],
}

REFERENCE_2D = {
    "spine_rear": (40.0, 40.0),
    "spine_front": (60.0, 40.0),
    "front_left_foot": (62.0, 60.0),
    "front_right_foot": (58.0, 60.0),
    "rear_left_foot": (42.0, 60.0),
    "rear_right_foot": (38.0, 60.0),
}


def _robot_and_rig() -> tuple[dict, dict]:
    sites = {
        f"site_{role}": {"pos": position}
        for role, position in NEUTRAL.items()
    }
    rig = {
        "schema": "qianji-key-site-map",
        "version": "0.1",
        "key_site_map": {
            role: f"site_{role}"
            for role in KEYPOINT_ROLES
        },
    }
    return {"name": "synthetic_cat", "sites": sites}, rig


def _point(x: float, y: float, *, valid: bool = True) -> dict:
    return {
        "x_px": x if valid else None,
        "y_px": y if valid else None,
        "confidence": 0.8,
        "valid": valid,
        "flags": [],
    }


def _frame(
    frame_idx: int,
    coordinates: dict[str, tuple[float, float]],
) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp_s": frame_idx / 10.0,
        "keypoints": {
            role: _point(*coordinates[role])
            for role in KEYPOINT_ROLES
        },
    }


def _trajectory(
    frames: list[dict] | None = None,
    *,
    identity_anchor: int | None = 0,
) -> dict:
    frames = frames or [_frame(0, REFERENCE_2D)]
    result = {
        "schema": "qianji.keypoint_trajectory_2d",
        "schema_version": "1.2.0",
        "coordinate_system": "image_pixels_top_left_origin_x_right_y_down",
        "video": {
            "width": 100,
            "height": 100,
            "fps": 10.0,
            "frame_count": len(frames),
        },
        "frames": frames,
    }
    if identity_anchor is not None:
        result["identity_anchor"] = {"frame_idx": identity_anchor}
    return result


def _transform_2d(
    coordinates: dict[str, tuple[float, float]],
    *,
    scale: float,
    angle_degrees: float,
    translation: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    angle = math.radians(angle_degrees)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    transformed = {}
    for role, coordinate in coordinates.items():
        value = scale * rotation @ np.asarray(coordinate) + translation
        transformed[role] = (float(value[0]), float(value[1]))
    return transformed


def _position(result, frame_idx: int, role: str) -> np.ndarray:
    return np.asarray(
        result.motion["frames"][frame_idx]["keypoints"][role][:3],
        dtype=float,
    )


def test_reference_pose_maps_exactly_to_neutral_rig() -> None:
    robot, rig = _robot_and_rig()

    result = lift_trajectory(_trajectory(), robot, rig)

    assert result.motion["schema"] == "qianji-keypoint-trajectory-v1"
    assert result.motion["fps"] == 10.0
    for role in KEYPOINT_ROLES:
        np.testing.assert_allclose(_position(result, 0, role), NEUTRAL[role])
        assert result.motion["frames"][0]["keypoints"][role][3] == 0.8


def test_camera_translation_rotation_and_zoom_are_removed() -> None:
    robot, rig = _robot_and_rig()
    transformed = _transform_2d(
        REFERENCE_2D,
        scale=1.7,
        angle_degrees=31.0,
        translation=(123.0, -48.0),
    )
    trajectory = _trajectory(
        [_frame(0, REFERENCE_2D), _frame(1, transformed)]
    )

    result = lift_trajectory(trajectory, robot, rig)

    for role in KEYPOINT_ROLES:
        np.testing.assert_allclose(
            _position(result, 1, role),
            NEUTRAL[role],
            atol=1e-12,
        )


def test_foot_lift_maps_to_up_without_inferred_lateral_motion() -> None:
    robot, rig = _robot_and_rig()
    lifted = dict(REFERENCE_2D)
    lifted["front_left_foot"] = (62.0, 50.0)
    trajectory = _trajectory(
        [_frame(0, REFERENCE_2D), _frame(1, lifted)]
    )

    result = lift_trajectory(
        trajectory,
        robot,
        rig,
        motion_scale=1.0,
    )

    displacement = _position(result, 1, "front_left_foot") - np.asarray(
        NEUTRAL["front_left_foot"]
    )
    assert displacement[2] == pytest.approx(0.5)
    assert displacement[0] == pytest.approx(0.0)
    assert displacement[1] == pytest.approx(0.0)

    lateral = np.asarray(result.binding["basis"]["lateral"])
    for role in KEYPOINT_ROLES:
        delta = _position(result, 1, role) - np.asarray(NEUTRAL[role])
        assert float(delta @ lateral) == pytest.approx(0.0, abs=1e-12)


def test_invalid_point_and_invalid_spine_use_neutral_zero_confidence() -> None:
    robot, rig = _robot_and_rig()
    frames = [
        _frame(0, REFERENCE_2D),
        _frame(1, REFERENCE_2D),
        _frame(2, REFERENCE_2D),
    ]
    frames[1]["keypoints"]["rear_left_foot"] = _point(0.0, 0.0, valid=False)
    frames[2]["keypoints"]["spine_front"] = _point(0.0, 0.0, valid=False)

    result = lift_trajectory(_trajectory(frames), robot, rig)

    invalid_foot = result.motion["frames"][1]["keypoints"]["rear_left_foot"]
    np.testing.assert_allclose(invalid_foot[:3], NEUTRAL["rear_left_foot"])
    assert invalid_foot[3] == 0.0
    for role in KEYPOINT_ROLES:
        point = result.motion["frames"][2]["keypoints"][role]
        np.testing.assert_allclose(point[:3], NEUTRAL[role])
        assert point[3] == 0.0
    assert result.report["substitution_counts"] == {
        "invalid_keypoint": 1,
        "invalid_spine_frame": 6,
    }


def test_reference_selection_prefers_anchor_then_median_torso() -> None:
    frames = [
        _frame(0, REFERENCE_2D),
        _frame(1, REFERENCE_2D),
        _frame(2, REFERENCE_2D),
    ]
    frames[0]["keypoints"]["spine_front"]["x_px"] = 50.0
    frames[1]["keypoints"]["spine_front"]["x_px"] = 70.0
    frames[2]["keypoints"]["spine_front"]["x_px"] = 60.0

    assert select_reference_frame(_trajectory(frames, identity_anchor=1)) == 1
    assert select_reference_frame(_trajectory(frames, identity_anchor=None)) == 2


def test_rig_validation_rejects_missing_and_duplicate_sites() -> None:
    robot, rig = _robot_and_rig()
    missing = copy.deepcopy(rig)
    del missing["key_site_map"]["rear_right_foot"]
    with pytest.raises(ValueError, match="missing rig roles"):
        neutral_pose_from_rig(robot, missing)

    duplicate = copy.deepcopy(rig)
    duplicate["key_site_map"]["rear_right_foot"] = duplicate["key_site_map"][
        "rear_left_foot"
    ]
    with pytest.raises(ValueError, match="distinct"):
        neutral_pose_from_rig(robot, duplicate)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_lift_publishes_complete_hashed_artifact_set(
    tmp_path: Path,
) -> None:
    robot, rig = _robot_and_rig()
    trajectory_path = tmp_path / "trajectory.json"
    robot_path = tmp_path / "robot.json"
    rig_path = tmp_path / "rig.json"
    _write_json(trajectory_path, _trajectory())
    _write_json(robot_path, robot)
    _write_json(rig_path, rig)

    outputs = run_lift(
        trajectory_path=trajectory_path,
        robot_path=robot_path,
        rig_path=rig_path,
        output_dir=tmp_path / "lifted",
    )

    assert tuple(path.name for path in outputs) == (
        "keypoint_motion.json",
        "mesh_binding.json",
        "lift_report.json",
    )
    assert all(path.is_file() for path in outputs)
    binding = json.loads(outputs.binding.read_text(encoding="utf-8"))
    assert binding["sources"] == {
        "trajectory": {
            "path": str(trajectory_path.resolve()),
            "sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        },
        "robot": {
            "path": str(robot_path.resolve()),
            "sha256": hashlib.sha256(robot_path.read_bytes()).hexdigest(),
        },
        "rig": {
            "path": str(rig_path.resolve()),
            "sha256": hashlib.sha256(rig_path.read_bytes()).hexdigest(),
        },
    }


def test_run_lift_refuses_overwrite_and_missing_sources(tmp_path: Path) -> None:
    robot, rig = _robot_and_rig()
    trajectory_path = tmp_path / "trajectory.json"
    robot_path = tmp_path / "robot.json"
    rig_path = tmp_path / "rig.json"
    _write_json(trajectory_path, _trajectory())
    _write_json(robot_path, robot)
    _write_json(rig_path, rig)
    arguments = {
        "trajectory_path": trajectory_path,
        "robot_path": robot_path,
        "rig_path": rig_path,
        "output_dir": tmp_path / "lifted",
    }

    run_lift(**arguments)
    with pytest.raises(FileExistsError, match="output already exists"):
        run_lift(**arguments)

    with pytest.raises(FileNotFoundError, match="trajectory source"):
        run_lift(
            **{
                **arguments,
                "trajectory_path": tmp_path / "missing.json",
                "output_dir": tmp_path / "other",
            }
        )


def test_run_lift_discards_staging_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, rig = _robot_and_rig()
    trajectory_path = tmp_path / "trajectory.json"
    robot_path = tmp_path / "robot.json"
    rig_path = tmp_path / "rig.json"
    _write_json(trajectory_path, _trajectory())
    _write_json(robot_path, robot)
    _write_json(rig_path, rig)

    def mutate_source(_path: Path, _payload: dict) -> None:
        trajectory_path.write_text("changed", encoding="utf-8")

    monkeypatch.setattr(
        "qianji_animal_motion.lift_cli._write_json",
        mutate_source,
    )
    output_dir = tmp_path / "lifted"
    with pytest.raises(RuntimeError, match="trajectory source changed"):
        run_lift(
            trajectory_path=trajectory_path,
            robot_path=robot_path,
            rig_path=rig_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".lifted.staging-*"))
