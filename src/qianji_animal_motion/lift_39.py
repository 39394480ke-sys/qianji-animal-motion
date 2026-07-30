"""Lift all 39 SuperAnimal landmarks into a mesh-derived neutral VGT frame."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from qianji_animal_motion.keypoints_39 import SUPERANIMAL_QUADRUPED_39
from qianji_animal_motion.lift_3d import neutral_pose_from_rig


@dataclass(frozen=True)
class Lift39Result:
    neutral_landmarks: dict
    motion: dict
    report: dict


def _spine_frame(frame: dict) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        rear = frame["keypoints"]["spine_rear"]
        front = frame["keypoints"]["spine_front"]
    except (KeyError, TypeError) as error:
        raise ValueError("corrected frame is missing spine roles") from error
    if rear.get("valid") is not True or front.get("valid") is not True:
        raise ValueError("corrected frame must contain a valid spine")
    try:
        rear_xy = np.asarray([rear["x_px"], rear["y_px"]], dtype=float)
        front_xy = np.asarray([front["x_px"], front["y_px"]], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("corrected spine coordinates are malformed") from error
    if not np.isfinite(rear_xy).all() or not np.isfinite(front_xy).all():
        raise ValueError("corrected spine coordinates must be finite")
    difference = front_xy - rear_xy
    torso = float(np.linalg.norm(difference))
    if torso <= 1e-9:
        raise ValueError("corrected spine torso length must be positive")
    forward = difference / torso
    up = np.asarray([forward[1], -forward[0]], dtype=float)
    if up[1] > 0.0:
        up = -up
    return (rear_xy + front_xy) / 2.0, np.stack([forward, up]), torso


def _neutral_basis(
    robot: dict,
    rig: dict,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    neutral = neutral_pose_from_rig(robot, rig)
    origin = (neutral["spine_rear"] + neutral["spine_front"]) / 2.0
    difference = neutral["spine_front"] - neutral["spine_rear"]
    torso = float(np.linalg.norm(difference))
    if torso <= 1e-9:
        raise ValueError("neutral VGT spine length must be positive")
    forward = difference / torso
    up = np.asarray([0.0, 0.0, 1.0]) - forward[2] * forward
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-9:
        raise ValueError("neutral VGT spine cannot be parallel to global up")
    up /= up_norm
    lateral = np.cross(forward, up)
    lateral /= np.linalg.norm(lateral)
    left_values = [
        float((neutral[role] - origin) @ lateral)
        for role in ("front_left_foot", "rear_left_foot")
    ]
    right_values = [
        float((neutral[role] - origin) @ lateral)
        for role in ("front_right_foot", "rear_right_foot")
    ]
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    half_width = 0.5 * abs(left_mean - right_mean)
    if half_width <= 1e-9:
        raise ValueError("neutral VGT rig must distinguish left and right")
    left_offset = math.copysign(half_width, left_mean - right_mean)
    right_offset = -left_offset
    return (
        neutral,
        origin,
        forward,
        up,
        lateral,
        torso,
        left_offset,
        right_offset,
    )


def _validate_inputs(
    trajectory_39: dict,
    corrected_spine: dict,
    reference_frame: int,
) -> None:
    if trajectory_39.get("schema") != "qianji.keypoint_trajectory_2d_39":
        raise ValueError("unsupported 39-point trajectory schema")
    if corrected_spine.get("schema") != "qianji.keypoint_trajectory_2d":
        raise ValueError("unsupported corrected spine schema")
    frames_39 = trajectory_39.get("frames")
    spine_frames = corrected_spine.get("frames")
    if not isinstance(frames_39, list) or not isinstance(spine_frames, list):
        raise ValueError("trajectories must contain frame lists")
    if not frames_39 or len(frames_39) != len(spine_frames):
        raise ValueError("39-point and corrected spine frame counts must match")
    if not 0 <= reference_frame < len(frames_39):
        raise ValueError("reference frame is outside the trajectory")
    if tuple(trajectory_39.get("roles", ())) != SUPERANIMAL_QUADRUPED_39:
        raise ValueError("39-point trajectory roles are missing or reordered")
    for frame_idx, (frame_39, spine_frame) in enumerate(
        zip(frames_39, spine_frames, strict=True)
    ):
        if (
            frame_39.get("frame_idx") != frame_idx
            or spine_frame.get("frame_idx") != frame_idx
        ):
            raise ValueError("trajectory frames must be contiguous from zero")
        if tuple(frame_39.get("keypoints", {})) != SUPERANIMAL_QUADRUPED_39:
            raise ValueError(f"frame {frame_idx} does not contain all 39 roles")
        _spine_frame(spine_frame)


def _side(role: str) -> str:
    if "left" in role:
        return "left"
    if "right" in role:
        return "right"
    return "center"


def _raw_xy(point: dict, role: str) -> np.ndarray:
    try:
        value = np.asarray(
            [point["raw_x_px"], point["raw_y_px"]],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"role {role!r} is missing finite raw coordinates") from error
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError(f"role {role!r} is missing finite raw coordinates")
    return value


def build_neutral_landmarks_39(
    trajectory_39: dict,
    corrected_spine: dict,
    robot: dict,
    rig: dict,
    *,
    reference_frame: int,
) -> dict:
    """Construct deterministic neutral 3D landmarks from the reference image."""
    _validate_inputs(trajectory_39, corrected_spine, reference_frame)
    (
        _neutral,
        origin_3d,
        forward_3d,
        up_3d,
        lateral_3d,
        torso_3d,
        left_offset,
        right_offset,
    ) = _neutral_basis(robot, rig)
    origin_2d, basis_2d, torso_2d = _spine_frame(
        corrected_spine["frames"][reference_frame]
    )
    reference_points = trajectory_39["frames"][reference_frame]["keypoints"]
    landmarks = {}
    for role in SUPERANIMAL_QUADRUPED_39:
        point = reference_points[role]
        local = basis_2d @ (_raw_xy(point, role) - origin_2d) / torso_2d
        side = _side(role)
        lateral_offset = (
            left_offset
            if side == "left"
            else right_offset
            if side == "right"
            else 0.0
        )
        position = (
            origin_3d
            + torso_3d * (local[0] * forward_3d + local[1] * up_3d)
            + lateral_offset * lateral_3d
        )
        landmarks[role] = {
            "xyz": position.tolist(),
            "side": side,
            "anatomy_applicable": "antler" not in role,
            "reference_confidence": float(point["confidence"]),
            "reference_valid": bool(point["valid"]),
            "reference_body_coordinate": local.tolist(),
            "construction": "reference_2d_body_frame_plus_vgt_lateral_identity",
        }
    return {
        "schema": "qianji.neutral_landmarks_39",
        "schema_version": "0.1.0",
        "reference_frame": reference_frame,
        "origin": origin_3d.tolist(),
        "basis": {
            "forward": forward_3d.tolist(),
            "up": up_3d.tolist(),
            "lateral": lateral_3d.tolist(),
        },
        "neutral_torso_length": torso_3d,
        "lateral_offsets": {
            "left": left_offset,
            "center": 0.0,
            "right": right_offset,
        },
        "landmarks": landmarks,
        "scientific_limits": {
            "metric_depth_observed": False,
            "camera_calibrated": False,
            "global_translation_preserved": False,
        },
    }


def lift_39point_trajectory(
    trajectory_39: dict,
    corrected_spine: dict,
    neutral_landmarks: dict,
    *,
    motion_scale: float,
) -> Lift39Result:
    """Transfer observable longitudinal/vertical changes to neutral landmarks."""
    reference_frame = neutral_landmarks.get("reference_frame")
    if not isinstance(reference_frame, int):
        raise ValueError("neutral landmarks must name an integer reference frame")
    _validate_inputs(trajectory_39, corrected_spine, reference_frame)
    if not math.isfinite(motion_scale) or motion_scale < 0.0:
        raise ValueError("motion scale must be finite and non-negative")
    if tuple(neutral_landmarks.get("landmarks", {})) != SUPERANIMAL_QUADRUPED_39:
        raise ValueError("neutral landmark roles are missing or reordered")

    forward = np.asarray(neutral_landmarks["basis"]["forward"], dtype=float)
    up = np.asarray(neutral_landmarks["basis"]["up"], dtype=float)
    lateral = np.asarray(neutral_landmarks["basis"]["lateral"], dtype=float)
    torso_3d = float(neutral_landmarks["neutral_torso_length"])
    if not all(
        np.isfinite(value).all()
        for value in (forward, up, lateral)
    ):
        raise ValueError("neutral basis must be finite")

    reference_origin, reference_basis, reference_torso = _spine_frame(
        corrected_spine["frames"][reference_frame]
    )
    reference_local = {}
    reference_points = trajectory_39["frames"][reference_frame]["keypoints"]
    for role in SUPERANIMAL_QUADRUPED_39:
        reference_local[role] = (
            reference_basis
            @ (_raw_xy(reference_points[role], role) - reference_origin)
            / reference_torso
        )

    substitutions = []
    output_frames = []
    max_lateral_delta = 0.0
    for frame_idx, (frame, spine_frame) in enumerate(
        zip(
            trajectory_39["frames"],
            corrected_spine["frames"],
            strict=True,
        )
    ):
        origin, basis, torso = _spine_frame(spine_frame)
        output_points = {}
        for role in SUPERANIMAL_QUADRUPED_39:
            point = frame["keypoints"][role]
            landmark = neutral_landmarks["landmarks"][role]
            neutral = np.asarray(landmark["xyz"], dtype=float)
            reason: str | None = None
            if not landmark["anatomy_applicable"]:
                reason = "inapplicable_anatomy"
            elif point.get("valid") is not True:
                reason = "invalid_observation"
            if reason is None:
                local = basis @ (_raw_xy(point, role) - origin) / torso
                delta = local - reference_local[role]
                displacement = motion_scale * torso_3d * (
                    delta[0] * forward + delta[1] * up
                )
                position = neutral + displacement
                confidence = float(point["confidence"])
                max_lateral_delta = max(
                    max_lateral_delta,
                    abs(float(displacement @ lateral)),
                )
            else:
                position = neutral
                confidence = 0.0
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
                "frame_idx": frame_idx,
                "time": float(
                    frame.get(
                        "timestamp_s",
                        frame_idx / trajectory_39["video"]["fps"],
                    )
                ),
                "keypoints": output_points,
            }
        )

    counts = Counter(item["reason"] for item in substitutions)
    motion = {
        "schema": "qianji-keypoint-trajectory-39-v1",
        "schema_version": "0.1.0",
        "fps": float(trajectory_39["video"]["fps"]),
        "source": "body_relative_2_5d_retarget",
        "roles": list(SUPERANIMAL_QUADRUPED_39),
        "frames": output_frames,
    }
    report = {
        "schema": "qianji.keypoint_lift_39_report",
        "schema_version": "0.1.0",
        "reconstruction_kind": "body_relative_2_5d_retarget",
        "reference_frame": reference_frame,
        "motion_scale": float(motion_scale),
        "frame_count": len(output_frames),
        "role_count": len(SUPERANIMAL_QUADRUPED_39),
        "substitutions": substitutions,
        "substitution_counts": dict(sorted(counts.items())),
        "maximum_lateral_displacement": max_lateral_delta,
        "metric_depth_observed": False,
        "camera_calibrated": False,
        "global_translation_preserved": False,
        "interpolation_applied": False,
        "smoothing_applied": False,
    }
    return Lift39Result(
        neutral_landmarks=neutral_landmarks,
        motion=motion,
        report=report,
    )
