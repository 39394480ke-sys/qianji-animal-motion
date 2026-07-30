from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from qianji_animal_motion.keypoints_39 import SUPERANIMAL_QUADRUPED_39
from qianji_animal_motion.lift_39 import (
    build_neutral_landmarks_39,
    lift_39point_trajectory,
)
from qianji_animal_motion.lift_39_cli import run_lift_39
from qianji_animal_motion.lift_3d import KEYPOINT_ROLES


RIG_NEUTRAL = {
    "spine_rear": [-0.5, 0.0, 0.8],
    "spine_front": [0.5, 0.0, 0.8],
    "front_left_foot": [0.5, 0.2, 0.0],
    "front_right_foot": [0.5, -0.2, 0.0],
    "rear_left_foot": [-0.5, 0.2, 0.0],
    "rear_right_foot": [-0.5, -0.2, 0.0],
}


def _robot_rig() -> tuple[dict, dict]:
    sites = {
        f"s{index:03d}": {
            "pos": RIG_NEUTRAL[role],
            "kind": "boundary",
        }
        for index, role in enumerate(KEYPOINT_ROLES)
    }
    for index in range(6, 12):
        sites[f"s{index:03d}"] = {
            "pos": [-0.4 + 0.16 * (index - 6), 0.0, 0.5],
            "kind": "interior",
        }
    rig = {
        "schema": "qianji-key-site-map",
        "key_site_map": {
            role: f"s{index:03d}"
            for index, role in enumerate(KEYPOINT_ROLES)
        },
    }
    return {"name": "cat", "sites": sites, "rod_groups": []}, rig


def _point(x: float, y: float, *, valid: bool = True) -> dict:
    return {
        "x_px": x if valid else None,
        "y_px": y if valid else None,
        "raw_x_px": x,
        "raw_y_px": y,
        "confidence": 0.8,
        "valid": valid,
        "identity_corrected": False,
        "flags": [] if valid else ["low_confidence"],
    }


def _coordinates() -> dict[str, tuple[float, float]]:
    return {
        role: (25.0 + index, 20.0 + (index % 7) * 4.0)
        for index, role in enumerate(SUPERANIMAL_QUADRUPED_39)
    }


def _trajectory_39(
    frames: list[dict] | None = None,
) -> dict:
    coordinates = _coordinates()
    frames = frames or [
        {
            "frame_idx": 0,
            "timestamp_s": 0.0,
            "keypoints": {
                role: _point(*coordinates[role])
                for role in SUPERANIMAL_QUADRUPED_39
            },
        }
    ]
    return {
        "schema": "qianji.keypoint_trajectory_2d_39",
        "schema_version": "0.1.0",
        "video": {
            "width": 100,
            "height": 80,
            "fps": 30.0,
            "frame_count": len(frames),
        },
        "roles": list(SUPERANIMAL_QUADRUPED_39),
        "frames": frames,
    }


def _corrected_spine(
    spine_pairs: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> dict:
    spine_pairs = spine_pairs or [((40.0, 40.0), (60.0, 40.0))]
    frames = []
    for frame_idx, (rear, front) in enumerate(spine_pairs):
        keypoints = {}
        for role in KEYPOINT_ROLES:
            coordinate = rear if role == "spine_rear" else front
            keypoints[role] = {
                "x_px": coordinate[0],
                "y_px": coordinate[1],
                "confidence": 0.9,
                "valid": True,
            }
        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / 30.0,
                "keypoints": keypoints,
            }
        )
    return {
        "schema": "qianji.keypoint_trajectory_2d",
        "video": {
            "width": 100,
            "height": 80,
            "fps": 30.0,
            "frame_count": len(frames),
        },
        "identity_anchor": {"frame_idx": 0},
        "frames": frames,
    }


def _transform(
    point: tuple[float, float],
    *,
    scale: float,
    angle_degrees: float,
    translation: tuple[float, float],
) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    value = scale * rotation @ np.asarray(point) + translation
    return float(value[0]), float(value[1])


def test_builds_all_neutral_landmarks_with_lateral_identity() -> None:
    robot, rig = _robot_rig()

    neutral = build_neutral_landmarks_39(
        _trajectory_39(),
        _corrected_spine(),
        robot,
        rig,
        reference_frame=0,
    )

    assert tuple(neutral["landmarks"]) == SUPERANIMAL_QUADRUPED_39
    forward = np.asarray(neutral["basis"]["forward"])
    up = np.asarray(neutral["basis"]["up"])
    lateral = np.asarray(neutral["basis"]["lateral"])
    np.testing.assert_allclose(
        np.stack([forward, up, lateral]) @ np.stack([forward, up, lateral]).T,
        np.eye(3),
        atol=1e-12,
    )
    left = np.asarray(neutral["landmarks"]["front_left_knee"]["xyz"])
    right = np.asarray(neutral["landmarks"]["front_right_knee"]["xyz"])
    center = np.asarray(neutral["origin"])
    assert float((left - center) @ lateral) != 0.0
    assert float((left - center) @ lateral) == pytest.approx(
        -float((right - center) @ lateral)
    )
    nose = np.asarray(neutral["landmarks"]["nose"]["xyz"])
    assert float((nose - center) @ lateral) == pytest.approx(0.0)
    assert (
        neutral["landmarks"]["left_antler_end"]["anatomy_applicable"]
        is False
    )


def test_reference_and_camera_transformed_pose_map_to_neutral() -> None:
    robot, rig = _robot_rig()
    coordinates = _coordinates()
    transformed = {
        role: _transform(
            coordinate,
            scale=1.4,
            angle_degrees=25.0,
            translation=(70.0, -15.0),
        )
        for role, coordinate in coordinates.items()
    }
    frames = []
    for frame_idx, values in enumerate((coordinates, transformed)):
        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / 30.0,
                "keypoints": {
                    role: _point(*values[role])
                    for role in SUPERANIMAL_QUADRUPED_39
                },
            }
        )
    rear1 = _transform(
        (40.0, 40.0),
        scale=1.4,
        angle_degrees=25.0,
        translation=(70.0, -15.0),
    )
    front1 = _transform(
        (60.0, 40.0),
        scale=1.4,
        angle_degrees=25.0,
        translation=(70.0, -15.0),
    )
    trajectory = _trajectory_39(frames)
    spine = _corrected_spine([((40.0, 40.0), (60.0, 40.0)), (rear1, front1)])
    neutral = build_neutral_landmarks_39(
        trajectory,
        spine,
        robot,
        rig,
        reference_frame=0,
    )

    result = lift_39point_trajectory(
        trajectory,
        spine,
        neutral,
        motion_scale=1.0,
    )

    for frame in result.motion["frames"]:
        for role in SUPERANIMAL_QUADRUPED_39:
            np.testing.assert_allclose(
                frame["keypoints"][role][:3],
                neutral["landmarks"][role]["xyz"],
                atol=1e-12,
            )


def test_lift_moves_up_and_substitutes_invalid_without_lateral_motion() -> None:
    robot, rig = _robot_rig()
    coordinates = _coordinates()
    moved = dict(coordinates)
    x, y = moved["front_left_knee"]
    moved["front_left_knee"] = (x, y - 10.0)
    frames = []
    for frame_idx, values in enumerate((coordinates, moved)):
        points = {
            role: _point(*values[role])
            for role in SUPERANIMAL_QUADRUPED_39
        }
        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / 30.0,
                "keypoints": points,
            }
        )
    frames[1]["keypoints"]["tail_end"] = _point(
        *moved["tail_end"],
        valid=False,
    )
    trajectory = _trajectory_39(frames)
    spine = _corrected_spine(
        [((40.0, 40.0), (60.0, 40.0)), ((40.0, 40.0), (60.0, 40.0))]
    )
    neutral = build_neutral_landmarks_39(
        trajectory,
        spine,
        robot,
        rig,
        reference_frame=0,
    )

    result = lift_39point_trajectory(
        trajectory,
        spine,
        neutral,
        motion_scale=1.0,
    )

    knee_neutral = np.asarray(neutral["landmarks"]["front_left_knee"]["xyz"])
    knee = np.asarray(
        result.motion["frames"][1]["keypoints"]["front_left_knee"][:3]
    )
    up = np.asarray(neutral["basis"]["up"])
    lateral = np.asarray(neutral["basis"]["lateral"])
    assert float((knee - knee_neutral) @ up) == pytest.approx(0.5)
    assert float((knee - knee_neutral) @ lateral) == pytest.approx(0.0)
    tail = result.motion["frames"][1]["keypoints"]["tail_end"]
    np.testing.assert_allclose(
        tail[:3],
        neutral["landmarks"]["tail_end"]["xyz"],
    )
    assert tail[3] == 0.0
    assert result.report["substitution_counts"] == {
        "inapplicable_anatomy": 8,
        "invalid_observation": 1,
    }
    for frame in result.motion["frames"]:
        assert tuple(frame["keypoints"]) == SUPERANIMAL_QUADRUPED_39
    assert "interpolation" not in result.motion


def test_lift_39_cli_publishes_atomic_hashed_outputs(tmp_path: Path) -> None:
    robot, rig = _robot_rig()
    source_payloads = {
        "trajectory_39": _trajectory_39(),
        "corrected_spine": _corrected_spine(),
        "robot": robot,
        "rig": rig,
    }
    source_paths = {}
    for label, payload in source_payloads.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        source_paths[label] = path

    outputs = run_lift_39(
        trajectory_39_path=source_paths["trajectory_39"],
        corrected_spine_path=source_paths["corrected_spine"],
        robot_path=source_paths["robot"],
        rig_path=source_paths["rig"],
        output_dir=tmp_path / "lifted",
        reference_frame=0,
        motion_scale=0.1,
    )

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    neutral = json.loads(outputs.neutral.read_text(encoding="utf-8"))
    for label, path in source_paths.items():
        assert neutral["sources"][label]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    with pytest.raises(FileExistsError, match="output already exists"):
        run_lift_39(
            trajectory_39_path=source_paths["trajectory_39"],
            corrected_spine_path=source_paths["corrected_spine"],
            robot_path=source_paths["robot"],
            rig_path=source_paths["rig"],
            output_dir=tmp_path / "lifted",
            reference_frame=0,
            motion_scale=0.1,
        )
