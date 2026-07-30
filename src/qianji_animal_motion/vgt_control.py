"""Map 39 observed landmarks onto six QianJi structural control sites."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from qianji_animal_motion.lift_3d import KEYPOINT_ROLES, neutral_pose_from_rig


CONTROL_OBSERVATIONS = {
    "spine_front": ("back_base", "neck_base"),
    "spine_rear": ("back_end", "tail_base"),
    "front_left_foot": ("front_left_paw", None),
    "front_right_foot": ("front_right_paw", None),
    "rear_left_foot": ("back_left_paw", None),
    "rear_right_foot": ("back_right_paw", None),
}


def _finite_vector(value: Any, *, dimensions: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite numbers") from error
    if vector.shape != (dimensions,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {dimensions} finite numbers")
    return vector


def _landmark_position(neutral_landmarks: dict, role: str) -> np.ndarray:
    landmarks = neutral_landmarks.get("landmarks")
    if not isinstance(landmarks, dict) or role not in landmarks:
        raise ValueError(f"neutral landmarks are missing role {role!r}")
    item = landmarks[role]
    if not isinstance(item, dict):
        raise ValueError(f"neutral landmark {role!r} must be an object")
    return _finite_vector(
        item.get("xyz"),
        dimensions=3,
        label=f"neutral landmark {role!r}",
    )


def _site_positions(robot: dict) -> dict[str, np.ndarray]:
    sites = robot.get("sites")
    if not isinstance(sites, dict):
        raise ValueError("robot sites must be an object")
    output = {}
    for name, item in sites.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError("robot sites must have non-empty names and objects")
        output[name] = _finite_vector(
            item.get("pos"),
            dimensions=3,
            label=f"robot site {name!r} position",
        )
    return output


def build_vgt_control_map(
    robot: dict,
    rig: dict,
    neutral_landmarks: dict,
) -> dict:
    """Declare the observation-to-control boundary for the six QianJi targets."""
    neutral_sites = neutral_pose_from_rig(robot, rig)
    site_map = rig["key_site_map"]
    controls = {}
    for role in KEYPOINT_ROLES:
        primary, fallback = CONTROL_OBSERVATIONS[role]
        controls[role] = {
            "site": site_map[role],
            "observation_primary": primary,
            "observation_fallback": fallback,
            "neutral_site_xyz": neutral_sites[role].tolist(),
            "neutral_primary_xyz": _landmark_position(
                neutral_landmarks,
                primary,
            ).tolist(),
            "transfer": "observation_displacement_plus_neutral_site",
        }
        if fallback is not None:
            controls[role]["neutral_fallback_xyz"] = _landmark_position(
                neutral_landmarks,
                fallback,
            ).tolist()
    return {
        "schema": "qianji.vgt_control_map",
        "schema_version": "0.1.0",
        "observation_role_count": len(neutral_landmarks.get("landmarks", {})),
        "control_role_count": len(KEYPOINT_ROLES),
        "controls": controls,
        "temporal_interpolation": False,
    }


def _motion_point(frame: dict, role: str) -> tuple[np.ndarray, float] | None:
    points = frame.get("keypoints")
    if not isinstance(points, dict) or role not in points:
        raise ValueError(f"39-point motion frame is missing role {role!r}")
    value = points[role]
    try:
        vector = np.asarray(value[:3], dtype=float)
        confidence = float(value[3])
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(f"motion role {role!r} must contain XYZC") from error
    if (
        vector.shape != (3,)
        or not np.isfinite(vector).all()
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError(f"motion role {role!r} must contain finite XYZC")
    if confidence <= 0.0:
        return None
    return vector, confidence


def build_target_control_motion(
    motion_39: dict,
    neutral_landmarks: dict,
    robot: dict,
    control_map: dict,
) -> tuple[dict, dict]:
    """Convert 39-point landmark displacement into six QianJi site targets."""
    if motion_39.get("schema") != "qianji-keypoint-trajectory-39-v1":
        raise ValueError("unsupported 39-point motion schema")
    if control_map.get("schema") != "qianji.vgt_control_map":
        raise ValueError("unsupported VGT control map schema")
    frames = motion_39.get("frames")
    controls = control_map.get("controls")
    if not isinstance(frames, list) or not frames:
        raise ValueError("39-point motion must contain frames")
    if not isinstance(controls, dict) or tuple(controls) != KEYPOINT_ROLES:
        raise ValueError("control map roles are missing or reordered")
    sites = _site_positions(robot)
    try:
        fps = float(motion_39["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("39-point motion fps is malformed") from error
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("39-point motion fps must be positive")

    fallbacks = []
    substitutions = []
    output_frames = []
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"motion frame {frame_idx} must be an object")
        output_points = {}
        for control_role in KEYPOINT_ROLES:
            item = controls[control_role]
            if not isinstance(item, dict):
                raise ValueError(f"control {control_role!r} must be an object")
            site_name = item.get("site")
            if site_name not in sites:
                raise ValueError(
                    f"control {control_role!r} references missing site {site_name!r}"
                )
            primary = item.get("observation_primary")
            fallback = item.get("observation_fallback")
            if not isinstance(primary, str):
                raise ValueError(f"control {control_role!r} has no primary")
            selected = _motion_point(frame, primary)
            selected_role = primary
            if selected is None and fallback is not None:
                if not isinstance(fallback, str):
                    raise ValueError(
                        f"control {control_role!r} fallback must be a role or null"
                    )
                selected = _motion_point(frame, fallback)
                selected_role = fallback
                if selected is not None:
                    fallbacks.append(
                        {
                            "frame_idx": frame_idx,
                            "control_role": control_role,
                            "invalid_primary": primary,
                            "used_observation": fallback,
                        }
                    )
            if selected is None:
                position = sites[site_name]
                confidence = 0.0
                substitutions.append(
                    {
                        "frame_idx": frame_idx,
                        "control_role": control_role,
                        "reason": "no_valid_observation",
                    }
                )
            else:
                observed, confidence = selected
                neutral_observed = _landmark_position(
                    neutral_landmarks,
                    selected_role,
                )
                position = sites[site_name] + observed - neutral_observed
            output_points[control_role] = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(confidence),
            ]
        try:
            time = float(frame.get("time", frame_idx / fps))
        except (TypeError, ValueError) as error:
            raise ValueError(f"motion frame {frame_idx} time is malformed") from error
        if not math.isfinite(time):
            raise ValueError(f"motion frame {frame_idx} time must be finite")
        output_frames.append({"time": time, "keypoints": output_points})

    target = {
        "schema": "qianji-keypoint-trajectory-v1",
        "fps": fps,
        "source": "39-point body-relative 2.5D VGT control adapter",
        "frames": output_frames,
    }
    report = {
        "schema": "qianji.vgt_control_transfer_report",
        "schema_version": "0.1.0",
        "frame_count": len(frames),
        "control_role_count": len(KEYPOINT_ROLES),
        "fallbacks": fallbacks,
        "neutral_substitutions": substitutions,
        "temporal_interpolation": False,
    }
    return target, report


def build_motion_informed_rig(
    robot: dict,
    neutral_landmarks: dict,
) -> dict:
    """Find the exact minimum-cost distinct site assignment for six controls."""
    sites = _site_positions(robot)
    if len(sites) < len(KEYPOINT_ROLES):
        raise ValueError("robot must contain at least six sites")
    site_names = tuple(sorted(sites))
    targets = tuple(
        _landmark_position(
            neutral_landmarks,
            CONTROL_OBSERVATIONS[role][0],
        )
        for role in KEYPOINT_ROLES
    )
    costs = np.asarray(
        [
            [
                float(np.linalg.norm(target - sites[site_name]))
                for site_name in site_names
            ]
            for target in targets
        ],
        dtype=float,
    )

    # Dynamic programming evaluates every distinct assignment exactly while
    # retaining the lexicographically smallest tuple for equal total cost.
    states: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    for role_index in range(len(KEYPOINT_ROLES)):
        next_states: dict[int, tuple[float, tuple[str, ...]]] = {}
        for mask, (cost, assignment) in states.items():
            for site_index, site_name in enumerate(site_names):
                bit = 1 << site_index
                if mask & bit:
                    continue
                candidate = (
                    cost + float(costs[role_index, site_index]),
                    assignment + (site_name,),
                )
                previous = next_states.get(mask | bit)
                if previous is None or candidate < previous:
                    next_states[mask | bit] = candidate
        states = next_states
    total_cost, assignment = min(states.values())
    return {
        "schema": "qianji-key-site-map",
        "version": "0.1",
        "source": "motion_informed_neutral_landmarks",
        "key_site_map": dict(zip(KEYPOINT_ROLES, assignment, strict=True)),
        "assignment": {
            "method": "exact_minimum_total_euclidean_distance",
            "distinct_sites": True,
            "role_order": list(KEYPOINT_ROLES),
            "site_tuple": list(assignment),
            "distance_by_role": {
                role: float(
                    np.linalg.norm(
                        targets[index] - sites[assignment[index]]
                    )
                )
                for index, role in enumerate(KEYPOINT_ROLES)
            },
            "total_distance": float(total_cost),
        },
    }


def apply_contraction_range(robot: dict, fraction: float) -> dict:
    """Return a robot copy permitting a bounded fraction of rod contraction."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 0.5:
        raise ValueError("contraction fraction must be finite and in [0, 0.5]")
    output = copy.deepcopy(robot)
    rods = output.get("rod_groups")
    if not isinstance(rods, list) or not rods:
        raise ValueError("robot rod_groups must be a non-empty list")
    mode = f"permitted_contraction_{fraction:.3f}"
    for index, rod in enumerate(rods):
        if not isinstance(rod, dict) or not isinstance(
            rod.get("constraint"),
            dict,
        ):
            raise ValueError(f"rod {index} must contain a constraint object")
        constraint = rod["constraint"]
        try:
            current = float(constraint["effective_current_length"])
            maximum = float(constraint["effective_max_length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"rod {index} constraint lengths are malformed"
            ) from error
        if (
            not math.isfinite(current)
            or not math.isfinite(maximum)
            or current <= 0.0
            or maximum < current
        ):
            raise ValueError(f"rod {index} constraint lengths are invalid")
        constraint["mode"] = mode
        constraint["effective_min_length"] = (1.0 - fraction) * current
    metadata = output.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("robot metadata must be an object")
    metadata["permitted_contraction_fraction"] = float(fraction)
    metadata["constraint_experiment"] = mode
    return output
