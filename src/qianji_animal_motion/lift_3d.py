"""Retarget six image-space quadruped points onto a neutral QianJi rig."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


KEYPOINT_ROLES = (
    "spine_front",
    "spine_rear",
    "front_left_foot",
    "front_right_foot",
    "rear_left_foot",
    "rear_right_foot",
)

_SPINE_ROLES = ("spine_rear", "spine_front")
_FOOT_ROLES = (
    "front_left_foot",
    "front_right_foot",
    "rear_left_foot",
    "rear_right_foot",
)
TRAJECTORY_SCHEMA = "qianji.keypoint_trajectory_2d"
SUPPORTED_TRAJECTORY_VERSIONS = frozenset({"1.2.0", "1.3.0"})
IMAGE_COORDINATE_SYSTEM = "image_pixels_top_left_origin_x_right_y_down"
TIMESTAMP_ABSOLUTE_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class LiftResult:
    motion: dict
    binding: dict
    report: dict


def _finite_vector(value: Any, *, dimensions: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite numbers") from error
    if vector.shape != (dimensions,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {dimensions} finite numbers")
    return vector


def neutral_pose_from_rig(
    robot: dict,
    rig: dict,
) -> dict[str, np.ndarray]:
    """Resolve all six semantic roles to distinct finite robot sites."""
    site_map = rig.get("key_site_map")
    if not isinstance(site_map, dict):
        raise ValueError("rig key_site_map must be an object")
    missing = sorted(set(KEYPOINT_ROLES) - set(site_map))
    if missing:
        raise ValueError("missing rig roles: " + ", ".join(missing))

    site_names = [site_map[role] for role in KEYPOINT_ROLES]
    if any(not isinstance(name, str) or not name for name in site_names):
        raise ValueError("rig site names must be non-empty strings")
    if len(set(site_names)) != len(site_names):
        raise ValueError("the six rig roles must map to distinct sites")

    sites = robot.get("sites")
    if not isinstance(sites, dict):
        raise ValueError("robot sites must be an object")
    neutral = {}
    for role, site_name in zip(KEYPOINT_ROLES, site_names, strict=True):
        if site_name not in sites or not isinstance(sites[site_name], dict):
            raise ValueError(f"rig site {site_name!r} for {role} is missing")
        neutral[role] = _finite_vector(
            sites[site_name].get("pos"),
            dimensions=3,
            label=f"robot site {site_name!r} position",
        )
    return neutral


def _valid_point(point: Any) -> bool:
    if not isinstance(point, dict) or point.get("valid") is not True:
        return False
    try:
        coordinates = np.asarray([point["x_px"], point["y_px"]], dtype=float)
        confidence = float(point["confidence"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(coordinates).all()
        and math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
    )


def _spine_geometry(frame: dict) -> tuple[np.ndarray, np.ndarray, float]:
    points = frame.get("keypoints")
    if not isinstance(points, dict):
        raise ValueError("frame keypoints must be an object")
    if not all(_valid_point(points.get(role)) for role in _SPINE_ROLES):
        raise ValueError("frame does not have a valid spine")

    rear = np.asarray(
        [
            points["spine_rear"]["x_px"],
            points["spine_rear"]["y_px"],
        ],
        dtype=float,
    )
    front = np.asarray(
        [
            points["spine_front"]["x_px"],
            points["spine_front"]["y_px"],
        ],
        dtype=float,
    )
    difference = front - rear
    torso_length = float(np.linalg.norm(difference))
    if not math.isfinite(torso_length) or torso_length <= 1e-9:
        raise ValueError("frame spine must have positive torso length")
    forward = difference / torso_length
    up = np.asarray([forward[1], -forward[0]], dtype=float)
    valid_feet = [
        np.asarray([points[role]["x_px"], points[role]["y_px"]], dtype=float)
        for role in _FOOT_ROLES
        if _valid_point(points.get(role))
    ]
    if valid_feet:
        ventral = np.mean(valid_feet, axis=0) - (rear + front) / 2.0
        ventral_alignment = float(up @ ventral)
        if ventral_alignment > 1e-6 * torso_length:
            up = -up
        elif abs(ventral_alignment) <= 1e-6 * torso_length and up[1] > 0.0:
            up = -up
    elif up[1] > 0.0:
        up = -up
    return (rear + front) / 2.0, np.stack([forward, up]), torso_length


def _validate_trajectory(trajectory: dict) -> list[dict]:
    if trajectory.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError("unsupported trajectory schema")
    if trajectory.get("schema_version") not in SUPPORTED_TRAJECTORY_VERSIONS:
        raise ValueError("unsupported trajectory schema_version")
    if trajectory.get("coordinate_system") != IMAGE_COORDINATE_SYSTEM:
        raise ValueError(
            "trajectory coordinate_system must be "
            f"{IMAGE_COORDINATE_SYSTEM!r}"
        )
    video = trajectory.get("video")
    frames = trajectory.get("frames")
    if not isinstance(video, dict) or not isinstance(frames, list) or not frames:
        raise ValueError("trajectory must contain video metadata and frames")
    try:
        fps = float(video["fps"])
        frame_count = int(video["frame_count"])
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("video metadata is malformed") from error
    if not math.isfinite(fps) or fps <= 0.0 or width <= 0 or height <= 0:
        raise ValueError("video dimensions and fps must be positive")
    if frame_count != len(frames):
        raise ValueError("video frame_count does not match frames")

    timestamps = []
    for expected_idx, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_idx") != expected_idx:
            raise ValueError("trajectory frames must be contiguous from zero")
        try:
            timestamp = float(frame["timestamp_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"frame {expected_idx} timestamp_s must be finite"
            ) from error
        if not math.isfinite(timestamp):
            raise ValueError(
                f"frame {expected_idx} timestamp_s must be finite"
            )
        timestamps.append(timestamp)
        points = frame.get("keypoints")
        if not isinstance(points, dict):
            raise ValueError(f"frame {expected_idx} keypoints must be an object")
        missing = sorted(set(KEYPOINT_ROLES) - set(points))
        if missing:
            raise ValueError(
                f"frame {expected_idx} is missing roles: " + ", ".join(missing)
            )
        for role in KEYPOINT_ROLES:
            point = points[role]
            if not isinstance(point, dict) or not isinstance(
                point.get("valid"), bool
            ):
                raise ValueError(
                    f"frame {expected_idx} {role} valid must be boolean"
                )
            if point["valid"] and not _valid_point(point):
                raise ValueError(
                    f"frame {expected_idx} {role} has malformed valid data"
                )
    if any(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("trajectory timestamps must be strictly increasing")
    for frame_idx, timestamp in enumerate(timestamps):
        expected = frame_idx / fps
        if not math.isclose(
            timestamp,
            expected,
            rel_tol=0.0,
            abs_tol=TIMESTAMP_ABSOLUTE_TOLERANCE_S,
        ):
            raise ValueError(
                f"frame {frame_idx} timestamp_s must match frame_idx/fps"
            )
    return frames


def select_reference_frame(trajectory: dict) -> int:
    """Choose a valid manual anchor, otherwise the median-torso frame."""
    frames = _validate_trajectory(trajectory)
    anchor = trajectory.get("identity_anchor")
    if isinstance(anchor, dict) and "frame_idx" in anchor:
        frame_idx = anchor["frame_idx"]
        if not isinstance(frame_idx, int) or not 0 <= frame_idx < len(frames):
            raise ValueError("identity anchor frame is outside the trajectory")
        _spine_geometry(frames[frame_idx])
        return frame_idx

    candidates = []
    for frame_idx, frame in enumerate(frames):
        try:
            torso_length = _spine_geometry(frame)[2]
        except ValueError:
            continue
        candidates.append((frame_idx, torso_length))
    if not candidates:
        raise ValueError("trajectory has no frame with a valid spine")
    median = float(np.median([length for _, length in candidates]))
    return min(candidates, key=lambda item: (abs(item[1] - median), item[0]))[0]


def _local_coordinates(
    frame: dict,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, float]]:
    origin, basis, torso_length = _spine_geometry(frame)
    local = {}
    for role in KEYPOINT_ROLES:
        point = frame["keypoints"][role]
        if not _valid_point(point):
            continue
        coordinate = np.asarray([point["x_px"], point["y_px"]], dtype=float)
        local[role] = basis @ (coordinate - origin) / torso_length
    return local, (origin, basis, torso_length)


def _neutral_basis(
    neutral: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    difference = neutral["spine_front"] - neutral["spine_rear"]
    torso_length = float(np.linalg.norm(difference))
    if not math.isfinite(torso_length) or torso_length <= 1e-9:
        raise ValueError("neutral rig spine must have positive length")
    forward = difference / torso_length
    up = np.asarray([0.0, 0.0, 1.0]) - forward[2] * forward
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-9:
        raise ValueError("neutral rig spine cannot be parallel to global up")
    up /= up_norm
    lateral = np.cross(forward, up)
    lateral /= np.linalg.norm(lateral)
    return forward, up, lateral, torso_length


def lift_trajectory(
    trajectory: dict,
    robot: dict,
    rig: dict,
    *,
    reference_frame: int | None = None,
    motion_scale: float = 0.25,
) -> LiftResult:
    """Transfer observable body-relative 2D articulation to a neutral 3D rig."""
    frames = _validate_trajectory(trajectory)
    neutral = neutral_pose_from_rig(robot, rig)
    if not math.isfinite(motion_scale) or motion_scale < 0.0:
        raise ValueError("motion_scale must be finite and non-negative")

    if reference_frame is None:
        reference_frame = select_reference_frame(trajectory)
    if not isinstance(reference_frame, int) or not 0 <= reference_frame < len(
        frames
    ):
        raise ValueError("reference frame is outside the trajectory")
    reference_local, _ = _local_coordinates(frames[reference_frame])
    if set(reference_local) != set(KEYPOINT_ROLES):
        raise ValueError("reference frame must contain all six valid keypoints")

    forward, up, lateral, torso_length_3d = _neutral_basis(neutral)
    substitutions = []
    output_frames = []
    displacement_norms = []
    for frame_idx, frame in enumerate(frames):
        try:
            local, _ = _local_coordinates(frame)
            valid_spine = True
        except ValueError:
            local = {}
            valid_spine = False

        output_points = {}
        for role in KEYPOINT_ROLES:
            point = frame["keypoints"][role]
            if valid_spine and role in local:
                delta = local[role] - reference_local[role]
                displacement = (
                    motion_scale
                    * torso_length_3d
                    * (delta[0] * forward + delta[1] * up)
                )
                position = neutral[role] + displacement
                confidence = float(point["confidence"])
                displacement_norms.append(float(np.linalg.norm(displacement)))
            else:
                position = neutral[role]
                confidence = 0.0
                reason = (
                    "invalid_spine_frame" if not valid_spine else "invalid_keypoint"
                )
                substitutions.append(
                    {
                        "frame_idx": frame_idx,
                        "role": role,
                        "reason": reason,
                    }
                )
            output_points[role] = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                confidence,
            ]
        output_frames.append(
            {
                "time": float(frame["timestamp_s"]),
                "keypoints": output_points,
            }
        )

    site_map = rig["key_site_map"]
    basis = {
        "forward": forward.tolist(),
        "up": up.tolist(),
        "lateral": lateral.tolist(),
    }
    motion = {
        "schema": "qianji-keypoint-trajectory-v1",
        "fps": float(trajectory["video"]["fps"]),
        "source": "qianji-animal-motion body-relative 2.5D retarget",
        "frames": output_frames,
    }
    binding = {
        "schema": "qianji.mesh_keypoint_binding",
        "schema_version": "0.1.0",
        "robot_name": robot.get("name"),
        "key_site_map": {role: site_map[role] for role in KEYPOINT_ROLES},
        "neutral_sites": {
            role: neutral[role].tolist() for role in KEYPOINT_ROLES
        },
        "basis": basis,
        "neutral_torso_length": torso_length_3d,
    }
    counts = Counter(item["reason"] for item in substitutions)
    report = {
        "schema": "qianji.keypoint_lift_report",
        "schema_version": "0.1.0",
        "reconstruction_kind": "body_relative_2_5d_retarget",
        "metric_depth_observed": False,
        "camera_calibrated": False,
        "global_translation_preserved": False,
        "input_contract": {
            "schema": trajectory["schema"],
            "schema_version": trajectory["schema_version"],
            "coordinate_system": trajectory["coordinate_system"],
            "timestamp_basis": "frame_idx/fps",
        },
        "reference_frame": reference_frame,
        "motion_scale": float(motion_scale),
        "frame_count": len(frames),
        "substitutions": substitutions,
        "substitution_counts": dict(sorted(counts.items())),
        "displacement_norm": {
            "minimum": min(displacement_norms, default=0.0),
            "maximum": max(displacement_norms, default=0.0),
            "mean": (
                float(np.mean(displacement_norms))
                if displacement_norms
                else 0.0
            ),
        },
        "assumptions": [
            "2D motion is expressed in a per-frame spine-aligned frame",
            "the neutral mesh rig supplies unobserved lateral placement",
            "no metric camera depth is inferred",
        ],
    }
    return LiftResult(motion=motion, binding=binding, report=report)
