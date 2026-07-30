"""Export all SuperAnimal-Quadruped landmarks without silent repair."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qianji_animal_motion.semantic_mapping import (
    VideoInfo,
    resolve_leg_identities,
)


SUPERANIMAL_QUADRUPED_39 = (
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

_JOINTS = ("thai", "knee", "paw")


@dataclass(frozen=True)
class Observation39Result:
    trajectory: dict
    report: dict


def _validate_dataframe(
    dataframe: pd.DataFrame,
    video: VideoInfo,
    individual: str,
    confidence_threshold: float,
    anchor_frame: int,
) -> str:
    expected_levels = {"scorer", "individuals", "bodyparts", "coords"}
    if set(dataframe.columns.names) != expected_levels:
        raise ValueError(
            "H5 columns must contain scorer, individuals, bodyparts, and coords"
        )
    scorers = list(dict.fromkeys(dataframe.columns.get_level_values("scorer")))
    individuals = set(dataframe.columns.get_level_values("individuals"))
    bodyparts = list(
        dict.fromkeys(dataframe.columns.get_level_values("bodyparts"))
    )
    coords = set(dataframe.columns.get_level_values("coords"))
    if len(scorers) != 1:
        raise ValueError(f"expected one scorer, found {scorers}")
    if individual not in individuals:
        raise ValueError(f"individual {individual!r} not found")
    if tuple(bodyparts) != SUPERANIMAL_QUADRUPED_39:
        missing = sorted(set(SUPERANIMAL_QUADRUPED_39) - set(bodyparts))
        extra = sorted(set(bodyparts) - set(SUPERANIMAL_QUADRUPED_39))
        raise ValueError(
            "bodyparts must match SuperAnimal-Quadruped 39 exactly; "
            f"missing={missing}, extra={extra}"
        )
    if not {"x", "y", "likelihood"}.issubset(coords):
        raise ValueError("coordinates must contain x, y, and likelihood")
    if len(dataframe) != video.frame_count:
        raise ValueError(
            "video frame_count does not match prediction rows: "
            f"{video.frame_count} != {len(dataframe)}"
        )
    if (
        video.width <= 0
        or video.height <= 0
        or not math.isfinite(video.fps)
        or video.fps <= 0
    ):
        raise ValueError("video metadata must be positive and finite")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be within [0, 1]")
    if not 0 <= anchor_frame < len(dataframe):
        raise ValueError("anchor frame must be within the trajectory")
    return str(scorers[0])


def _bodypart_array(
    dataframe: pd.DataFrame,
    scorer: str,
    individual: str,
    role: str,
) -> np.ndarray:
    values = dataframe.loc[
        :,
        pd.IndexSlice[
            scorer,
            individual,
            role,
            ["x", "y", "likelihood"],
        ],
    ].to_numpy(dtype=float)
    if values.shape != (len(dataframe), 3) or not np.isfinite(values).all():
        raise ValueError(f"bodypart {role!r} must contain finite x/y/likelihood")
    if ((values[:, 2] < 0.0) | (values[:, 2] > 1.0)).any():
        raise ValueError(f"bodypart {role!r} likelihood must be within [0, 1]")
    return values


def _leg_chain(arrays: dict[str, np.ndarray], prefix: str, side: str) -> np.ndarray:
    return np.stack(
        [arrays[f"{prefix}_{side}_{joint}"] for joint in _JOINTS],
        axis=1,
    )


def _anchor_state(value: str, label: str) -> int:
    if value == "keep":
        return 0
    if value == "swap":
        return 1
    raise ValueError(f"{label} anchor must be 'keep' or 'swap'")


def _resolved_arrays(
    arrays: dict[str, np.ndarray],
    *,
    anchor_frame: int,
    front_anchor: str,
    rear_anchor: str,
    scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    resolved = {role: values.copy() for role, values in arrays.items()}
    identity_report = {}
    for source_prefix, report_name in (("front", "front"), ("back", "rear")):
        left = _leg_chain(arrays, source_prefix, "left")
        right = _leg_chain(arrays, source_prefix, "right")
        anchor_value = front_anchor if report_name == "front" else rear_anchor
        identity = resolve_leg_identities(
            left,
            right,
            scale=scale,
            anchor_frame=anchor_frame,
            anchor_state=_anchor_state(anchor_value, report_name),
        )
        for frame_idx, state in enumerate(identity.states):
            if state == 0:
                continue
            for joint in _JOINTS:
                left_role = f"{source_prefix}_left_{joint}"
                right_role = f"{source_prefix}_right_{joint}"
                resolved[left_role][frame_idx] = arrays[right_role][frame_idx]
                resolved[right_role][frame_idx] = arrays[left_role][frame_idx]
        identity_report[report_name] = {
            "anchor_assignment": anchor_value,
            "swapped_frames": np.flatnonzero(identity.states)
            .astype(int)
            .tolist(),
            "ambiguous_frames": np.flatnonzero(identity.ambiguous)
            .astype(int)
            .tolist(),
        }
    return resolved, identity_report


def _torso_scales(
    arrays: dict[str, np.ndarray],
    video: VideoInfo,
) -> tuple[np.ndarray, list[int]]:
    difference = arrays["back_base"][:, :2] - arrays["back_end"][:, :2]
    scales = np.linalg.norm(difference, axis=1)
    fallback = ~np.isfinite(scales) | (scales <= 1e-6)
    scales[fallback] = math.hypot(video.width, video.height)
    return scales, np.flatnonzero(fallback).astype(int).tolist()


def build_39point_observation(
    dataframe: pd.DataFrame,
    video: VideoInfo,
    *,
    individual: str = "animal0",
    confidence_threshold: float = 0.3,
    anchor_frame: int = 152,
    front_anchor: str = "keep",
    rear_anchor: str = "keep",
) -> Observation39Result:
    """Convert a SuperAnimal H5 dataframe into an explicit 39-role trajectory."""
    scorer = _validate_dataframe(
        dataframe,
        video,
        individual,
        confidence_threshold,
        anchor_frame,
    )
    arrays = {
        role: _bodypart_array(dataframe, scorer, individual, role)
        for role in SUPERANIMAL_QUADRUPED_39
    }
    torso_scales, torso_fallback_frames = _torso_scales(arrays, video)
    resolved, identity_report = _resolved_arrays(
        arrays,
        anchor_frame=anchor_frame,
        front_anchor=front_anchor,
        rear_anchor=rear_anchor,
        scale=float(np.median(torso_scales)),
    )
    corrected_frames = {
        report_name: set(section["swapped_frames"])
        for report_name, section in identity_report.items()
    }

    frames = []
    role_reports = {}
    for role in SUPERANIMAL_QUADRUPED_39:
        invalid_frames = []
        flags_counter: Counter[str] = Counter()
        role_points = []
        prefix = (
            "front"
            if role.startswith("front_")
            else "rear"
            if role.startswith("back_") and any(
                role.endswith(joint) for joint in _JOINTS
            )
            else None
        )
        values = resolved[role]
        for frame_idx, (x, y, confidence) in enumerate(values):
            flags = []
            if confidence < confidence_threshold:
                flags.append("low_confidence")
            if not 0.0 <= x < video.width or not 0.0 <= y < video.height:
                flags.append("out_of_bounds")
            if frame_idx > 0:
                step = float(np.linalg.norm(values[frame_idx, :2] - values[frame_idx - 1, :2]))
                if step > 0.75 * torso_scales[frame_idx]:
                    flags.append("temporal_jump")
            valid = not flags
            if not valid:
                invalid_frames.append(frame_idx)
                flags_counter.update(flags)
            identity_corrected = (
                prefix is not None
                and frame_idx in corrected_frames[prefix]
            )
            role_points.append(
                {
                    "x_px": float(x) if valid else None,
                    "y_px": float(y) if valid else None,
                    "raw_x_px": float(x),
                    "raw_y_px": float(y),
                    "confidence": float(confidence),
                    "valid": valid,
                    "identity_corrected": identity_corrected,
                    "flags": flags,
                }
            )
        role_reports[role] = {
            "invalid_frames": invalid_frames,
            "valid_frames": len(values) - len(invalid_frames),
            "flag_counts": dict(sorted(flags_counter.items())),
            "confidence": {
                "minimum": float(np.min(values[:, 2])),
                "median": float(np.median(values[:, 2])),
                "maximum": float(np.max(values[:, 2])),
            },
        }
        for frame_idx, point in enumerate(role_points):
            if len(frames) <= frame_idx:
                frames.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": frame_idx / video.fps,
                        "keypoints": {},
                    }
                )
            frames[frame_idx]["keypoints"][role] = point

    trajectory = {
        "schema": "qianji.keypoint_trajectory_2d_39",
        "schema_version": "0.1.0",
        "coordinate_system": "image_pixels_top_left_origin_x_right_y_down",
        "video": {
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frame_count": video.frame_count,
        },
        "roles": list(SUPERANIMAL_QUADRUPED_39),
        "identity_anchor": {
            "frame_idx": anchor_frame,
            "front_assignment": front_anchor,
            "rear_assignment": rear_anchor,
        },
        "quality_policy": {
            "confidence_threshold": confidence_threshold,
            "temporal_jump_torso_fraction": 0.75,
            "interpolation": "none",
            "smoothing": "none",
        },
        "frames": frames,
    }
    report = {
        "schema": "qianji.keypoint_39_quality_report",
        "schema_version": "0.1.0",
        "scorer": scorer,
        "individual": individual,
        "frame_count": video.frame_count,
        "role_count": len(SUPERANIMAL_QUADRUPED_39),
        "confidence_threshold": confidence_threshold,
        "torso_scale_fallback_frames": torso_fallback_frames,
        "identity": identity_report,
        "roles": role_reports,
        "interpolation_applied": False,
        "smoothing_applied": False,
    }
    return Observation39Result(trajectory=trajectory, report=report)
