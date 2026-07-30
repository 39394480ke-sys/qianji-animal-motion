"""Lift all 39 SuperAnimal landmarks into a mesh-derived neutral VGT frame."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from qianji_animal_motion.keypoints_39 import SUPERANIMAL_QUADRUPED_39
from qianji_animal_motion.lift_3d import (
    IMAGE_COORDINATE_SYSTEM,
    _validate_trajectory,
    neutral_pose_from_rig,
)


REFERENCE_SEARCH_RADIUS_FRAMES = 15


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
    if trajectory_39.get("schema_version") != "0.1.0":
        raise ValueError("unsupported 39-point trajectory schema_version")
    if trajectory_39.get("coordinate_system") != IMAGE_COORDINATE_SYSTEM:
        raise ValueError("unsupported 39-point trajectory coordinate_system")
    _validate_trajectory(corrected_spine)
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
    video_39 = trajectory_39.get("video")
    video_spine = corrected_spine.get("video")
    if not isinstance(video_39, dict) or not isinstance(video_spine, dict):
        raise ValueError("both trajectories must contain video metadata")
    for field in ("width", "height", "frame_count"):
        if video_39.get(field) != video_spine.get(field):
            raise ValueError(f"trajectory video {field} values must match")
    try:
        fps_39 = float(video_39["fps"])
        fps_spine = float(video_spine["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("trajectory video fps values must be finite") from error
    if (
        not math.isfinite(fps_39)
        or not math.isfinite(fps_spine)
        or not math.isclose(fps_39, fps_spine, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("trajectory video fps values must match")
    if video_39.get("frame_count") != len(frames_39):
        raise ValueError("39-point video frame_count does not match frames")

    lineage = trajectory_39.get("source_lineage")
    corrected_source = corrected_spine.get("source")
    if not isinstance(lineage, dict) or not isinstance(corrected_source, dict):
        raise ValueError("trajectory source lineage is required")
    for field in ("video_sha256", "predictions_sha256"):
        value = lineage.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"39-point source_lineage {field} is invalid")
        if value != corrected_source.get(field):
            raise ValueError(f"trajectory {field} values must match")
    corrected_parent = lineage.get("corrected_trajectory_sha256")
    if not isinstance(corrected_parent, str) or len(corrected_parent) != 64:
        raise ValueError(
            "39-point corrected_trajectory_sha256 parent is invalid"
        )

    anchor_39 = trajectory_39.get("identity_anchor")
    anchor_spine = corrected_spine.get("identity_anchor")
    if (
        not isinstance(anchor_39, dict)
        or anchor_39.get("frame_idx") != reference_frame
        or not isinstance(anchor_spine, dict)
        or anchor_spine.get("frame_idx") != reference_frame
    ):
        raise ValueError("identity anchor must match the reference frame")

    previous_time = -math.inf
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
        try:
            time_39 = float(frame_39["timestamp_s"])
            time_spine = float(spine_frame["timestamp_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"frame {frame_idx} timestamp must be finite"
            ) from error
        expected_time = frame_idx / fps_39
        if (
            not math.isfinite(time_39)
            or not math.isclose(
                time_39,
                time_spine,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                time_39,
                expected_time,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"frame {frame_idx} trajectory timestamp values must match"
            )
        if time_39 <= previous_time:
            raise ValueError("trajectory timestamps must be strictly increasing")
        previous_time = time_39
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


def _reference_frame_for_role(
    trajectory_39: dict,
    role: str,
    reference_frame: int,
) -> int | None:
    start = max(0, reference_frame - REFERENCE_SEARCH_RADIUS_FRAMES)
    stop = min(
        len(trajectory_39["frames"]),
        reference_frame + REFERENCE_SEARCH_RADIUS_FRAMES + 1,
    )
    candidates = sorted(
        range(start, stop),
        key=lambda frame_idx: (abs(frame_idx - reference_frame), frame_idx),
    )
    for frame_idx in candidates:
        point = trajectory_39["frames"][frame_idx]["keypoints"][role]
        if point.get("valid") is not True:
            continue
        try:
            _raw_xy(point, role)
            confidence = float(point["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(confidence) and 0.0 <= confidence <= 1.0:
            return frame_idx
    return None


def validate_neutral_landmarks_39(neutral: dict) -> dict:
    """Validate role anatomy, reference availability, and finite neutral XYZ."""
    if neutral.get("schema") != "qianji.neutral_landmarks_39":
        raise ValueError("unsupported neutral 39-landmark schema")
    if neutral.get("schema_version") != "0.1.0":
        raise ValueError("unsupported neutral 39-landmark schema_version")
    reference_frame = neutral.get("reference_frame")
    if not isinstance(reference_frame, int) or reference_frame < 0:
        raise ValueError("neutral reference_frame must be a non-negative integer")
    if (
        neutral.get("reference_search_radius_frames")
        != REFERENCE_SEARCH_RADIUS_FRAMES
    ):
        raise ValueError("neutral reference search radius is invalid")
    frame_by_role = neutral.get("reference_frame_by_role")
    landmarks = neutral.get("landmarks")
    if (
        not isinstance(frame_by_role, dict)
        or tuple(frame_by_role) != SUPERANIMAL_QUADRUPED_39
        or not isinstance(landmarks, dict)
        or tuple(landmarks) != SUPERANIMAL_QUADRUPED_39
    ):
        raise ValueError("neutral landmarks must contain the exact 39 roles")
    for name in ("forward", "up", "lateral"):
        vector = np.asarray(neutral.get("basis", {}).get(name), dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"neutral basis {name} must contain finite XYZ")
    try:
        torso = float(neutral["neutral_torso_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("neutral torso length is malformed") from error
    if not math.isfinite(torso) or torso <= 0.0:
        raise ValueError("neutral torso length must be positive")
    for role, item in landmarks.items():
        if not isinstance(item, dict):
            raise ValueError(f"neutral landmark {role!r} must be an object")
        xyz = np.asarray(item.get("xyz"), dtype=float)
        local = np.asarray(item.get("reference_body_coordinate"), dtype=float)
        if (
            xyz.shape != (3,)
            or local.shape != (2,)
            or not np.isfinite(xyz).all()
            or not np.isfinite(local).all()
        ):
            raise ValueError(f"neutral landmark {role!r} coordinates are invalid")
        expected_side = _side(role)
        if item.get("side") != expected_side:
            raise ValueError(f"neutral landmark {role!r} side is invalid")
        applicable = item.get("anatomy_applicable")
        expected_applicable = "antler" not in role
        if applicable is not expected_applicable:
            raise ValueError(f"neutral antler anatomy for {role!r} is invalid")
        available = item.get("available")
        if not isinstance(available, bool):
            raise ValueError(f"neutral landmark {role!r} availability is invalid")
        role_reference = item.get("reference_frame")
        if frame_by_role[role] != role_reference:
            raise ValueError(f"neutral landmark {role!r} reference frames differ")
        if available:
            if not isinstance(role_reference, int) or role_reference < 0:
                raise ValueError(
                    f"neutral landmark {role!r} reference frame is invalid"
                )
            confidence = item.get("reference_confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
                or item.get("reference_valid") is not True
            ):
                raise ValueError(
                    f"neutral landmark {role!r} reference confidence is invalid"
                )
        elif (
            role_reference is not None
            or item.get("reference_confidence") is not None
            or item.get("reference_valid") is not False
        ):
            raise ValueError(
                f"neutral landmark {role!r} unavailable reference is inconsistent"
            )
        if not expected_applicable and available:
            raise ValueError(f"neutral antler landmark {role!r} must be unavailable")
    limits = neutral.get("scientific_limits")
    if not isinstance(limits, dict):
        raise ValueError("neutral scientific_limits are required")
    for field in (
        "metric_depth_observed",
        "camera_calibrated",
        "global_translation_preserved",
    ):
        if field not in limits or limits[field] is not False:
            raise ValueError(f"neutral scientific limit {field} must be false")
    return {"roles": len(landmarks), "reference_frame": reference_frame}


def validate_lifted_39_motion(
    motion: dict,
    report: dict,
    *,
    observation: dict | None = None,
    neutral_landmarks: dict | None = None,
    expected_frames: int | None = None,
) -> dict:
    """Validate lifted XYZC, substitutions, timestamps, and explicit limits."""
    if motion.get("schema") != "qianji-keypoint-trajectory-39-v1":
        raise ValueError("unsupported lifted 39-point motion schema")
    if motion.get("schema_version") != "0.1.0":
        raise ValueError("unsupported lifted 39-point motion schema_version")
    if motion.get("source") != "body_relative_2_5d_retarget":
        raise ValueError("lifted 39-point motion source is invalid")
    if tuple(motion.get("roles", ())) != SUPERANIMAL_QUADRUPED_39:
        raise ValueError("lifted 39-point roles are missing or reordered")
    try:
        fps = float(motion["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("lifted 39-point fps is malformed") from error
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("lifted 39-point fps must be positive")
    frames = motion.get("frames")
    if (
        not isinstance(frames, list)
        or not frames
        or (expected_frames is not None and len(frames) != expected_frames)
    ):
        raise ValueError("lifted 39-point frame count is invalid")
    if (
        report.get("schema") != "qianji.keypoint_lift_39_report"
        or report.get("schema_version") != "0.1.0"
        or report.get("reconstruction_kind")
        != "body_relative_2_5d_retarget"
        or report.get("frame_count") != len(frames)
        or report.get("role_count") != len(SUPERANIMAL_QUADRUPED_39)
    ):
        raise ValueError("lifted 39-point report contract is invalid")
    for field in ("interpolation_applied", "smoothing_applied"):
        if field not in report or report[field] is not False:
            raise ValueError(f"lift report {field} must be explicit false")
    for field in (
        "metric_depth_observed",
        "camera_calibrated",
        "global_translation_preserved",
    ):
        if field not in report or report[field] is not False:
            raise ValueError(f"lift report {field} must be explicit false")
    try:
        motion_scale = float(report["motion_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("lift report motion_scale is malformed") from error
    if not math.isfinite(motion_scale) or motion_scale < 0.0:
        raise ValueError("lift report motion_scale is invalid")
    substitutions = report.get("substitutions")
    counts = report.get("substitution_counts")
    if not isinstance(substitutions, list) or not isinstance(counts, dict):
        raise ValueError("lift report substitutions are required")
    expected_substitutions = []
    if neutral_landmarks is not None:
        validate_neutral_landmarks_39(neutral_landmarks)
    if observation is not None:
        observation_frames = observation.get("frames")
        if not isinstance(observation_frames, list) or len(observation_frames) != len(
            frames
        ):
            raise ValueError("lifted motion observation frame count differs")
        if neutral_landmarks is not None:
            reference_frame = neutral_landmarks["reference_frame"]
            if (
                observation.get("identity_anchor", {}).get("frame_idx")
                != reference_frame
                or report.get("reference_frame") != reference_frame
            ):
                raise ValueError(
                    "lifted observation, neutral, and report reference frames differ"
                )
            for role in SUPERANIMAL_QUADRUPED_39:
                expected_reference = (
                    _reference_frame_for_role(
                        observation,
                        role,
                        reference_frame,
                    )
                    if "antler" not in role
                    else None
                )
                landmark = neutral_landmarks["landmarks"][role]
                if landmark["reference_frame"] != expected_reference:
                    raise ValueError(
                        f"neutral landmark {role!r} reference frame "
                        "does not match the observation"
                    )
                if expected_reference is not None:
                    observed = observation_frames[expected_reference][
                        "keypoints"
                    ][role]
                    if not math.isclose(
                        float(landmark["reference_confidence"]),
                        float(observed["confidence"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"neutral landmark {role!r} reference confidence "
                            "does not match the observation"
                        )
    previous_time = -math.inf
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_idx") != frame_idx:
            raise ValueError("lifted 39-point frames must be contiguous")
        try:
            time = float(frame["time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"lifted frame {frame_idx} time is malformed") from error
        if (
            not math.isfinite(time)
            or time <= previous_time
            or not math.isclose(
                time,
                frame_idx / fps,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(f"lifted frame {frame_idx} time is invalid")
        previous_time = time
        points = frame.get("keypoints")
        if not isinstance(points, dict) or tuple(points) != SUPERANIMAL_QUADRUPED_39:
            raise ValueError(f"lifted frame {frame_idx} roles differ")
        for role, raw_value in points.items():
            value = np.asarray(raw_value, dtype=float)
            if value.shape != (4,) or not np.isfinite(value).all():
                raise ValueError(
                    f"lifted frame {frame_idx} role {role!r} must contain finite XYZC"
                )
            if not 0.0 <= value[3] <= 1.0:
                raise ValueError(
                    f"lifted frame {frame_idx} role {role!r} confidence is invalid"
                )
            reason = None
            landmark = (
                neutral_landmarks["landmarks"][role]
                if neutral_landmarks is not None
                else None
            )
            if landmark is not None:
                if landmark["anatomy_applicable"] is not True:
                    reason = "inapplicable_anatomy"
                elif landmark["available"] is not True:
                    reason = "reference_unavailable"
                elif (
                    observation is not None
                    and observation["frames"][frame_idx]["keypoints"][role][
                        "valid"
                    ]
                    is not True
                ):
                    reason = "invalid_observation"
            if reason is not None:
                expected_substitutions.append(
                    {
                        "frame_idx": frame_idx,
                        "role": role,
                        "reason": reason,
                    }
                )
                if value[3] != 0.0 or not np.allclose(
                    value[:3],
                    np.asarray(landmark["xyz"], dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"lifted substitution for frame {frame_idx} role {role!r} "
                        "does not use neutral XYZ/zero confidence"
                    )
            elif observation is not None:
                observed_confidence = float(
                    observation["frames"][frame_idx]["keypoints"][role][
                        "confidence"
                    ]
                )
                if not math.isclose(
                    value[3],
                    observed_confidence,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"lifted frame {frame_idx} role {role!r} confidence differs"
                    )
            if "antler" in role and value[3] != 0.0:
                raise ValueError("lifted antler confidence must be zero")
    if neutral_landmarks is not None and observation is not None:
        if substitutions != expected_substitutions:
            raise ValueError("lift report substitutions disagree with inputs")
        expected_counts = Counter(
            item["reason"] for item in expected_substitutions
        )
        if counts != dict(sorted(expected_counts.items())):
            raise ValueError("lift report substitution counts disagree")
    return {
        "frames": len(frames),
        "roles": len(SUPERANIMAL_QUADRUPED_39),
        "substitutions": len(substitutions),
    }


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
    landmarks = {}
    reference_frame_by_role = {}
    for role in SUPERANIMAL_QUADRUPED_39:
        anatomy_applicable = "antler" not in role
        role_reference = (
            _reference_frame_for_role(
                trajectory_39,
                role,
                reference_frame,
            )
            if anatomy_applicable
            else None
        )
        reference_frame_by_role[role] = role_reference
        if role_reference is None:
            local = np.zeros(2, dtype=float)
            reference_confidence = None
        else:
            origin_2d, basis_2d, torso_2d = _spine_frame(
                corrected_spine["frames"][role_reference]
            )
            point = trajectory_39["frames"][role_reference]["keypoints"][role]
            local = (
                basis_2d @ (_raw_xy(point, role) - origin_2d) / torso_2d
            )
            reference_confidence = float(point["confidence"])
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
            "anatomy_applicable": anatomy_applicable,
            "available": role_reference is not None,
            "reference_frame": role_reference,
            "reference_confidence": reference_confidence,
            "reference_valid": role_reference is not None,
            "reference_body_coordinate": local.tolist(),
            "construction": (
                "reference_2d_body_frame_plus_vgt_lateral_identity"
                if role_reference is not None
                else "neutral_placeholder_without_observed_reference"
            ),
        }
    return {
        "schema": "qianji.neutral_landmarks_39",
        "schema_version": "0.1.0",
        "reference_frame": reference_frame,
        "reference_search_radius_frames": REFERENCE_SEARCH_RADIUS_FRAMES,
        "reference_frame_by_role": reference_frame_by_role,
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

    reference_local = {}
    for role in SUPERANIMAL_QUADRUPED_39:
        reference_local[role] = np.asarray(
            neutral_landmarks["landmarks"][role]["reference_body_coordinate"],
            dtype=float,
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
            elif landmark.get("available") is not True:
                reason = "reference_unavailable"
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
                "time": float(frame["timestamp_s"]),
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
