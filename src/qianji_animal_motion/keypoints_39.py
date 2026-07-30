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


def _finite_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_39point_observation(
    trajectory: dict,
    quality_report: dict | None = None,
    *,
    expected_frames: int | None = None,
    require_source_lineage: bool = False,
) -> dict:
    """Validate the complete semantic contract of a 39-point observation."""
    if trajectory.get("schema") != "qianji.keypoint_trajectory_2d_39":
        raise ValueError("unsupported 39-point observation schema")
    if trajectory.get("schema_version") != "0.1.0":
        raise ValueError("unsupported 39-point observation schema_version")
    if (
        trajectory.get("coordinate_system")
        != "image_pixels_top_left_origin_x_right_y_down"
    ):
        raise ValueError("unsupported 39-point observation coordinate system")
    if tuple(trajectory.get("roles", ())) != SUPERANIMAL_QUADRUPED_39:
        raise ValueError("39-point observation roles are missing or reordered")
    video = trajectory.get("video")
    frames = trajectory.get("frames")
    if not isinstance(video, dict) or not isinstance(frames, list) or not frames:
        raise ValueError("39-point observation must contain video and frames")
    try:
        width = int(video["width"])
        height = int(video["height"])
        fps = float(video["fps"])
        frame_count = int(video["frame_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("39-point video metadata is malformed") from error
    if (
        width <= 0
        or height <= 0
        or not math.isfinite(fps)
        or fps <= 0.0
        or frame_count != len(frames)
        or (expected_frames is not None and frame_count != expected_frames)
    ):
        raise ValueError("39-point video metadata does not match frames")
    anchor = trajectory.get("identity_anchor")
    if (
        not isinstance(anchor, dict)
        or not isinstance(anchor.get("frame_idx"), int)
        or not 0 <= anchor["frame_idx"] < frame_count
    ):
        raise ValueError("39-point identity_anchor is malformed")
    for field in ("front_assignment", "rear_assignment"):
        if anchor.get(field) not in {"keep", "swap"}:
            raise ValueError(f"identity_anchor {field} must be keep or swap")
    quality_policy = trajectory.get("quality_policy")
    if (
        not isinstance(quality_policy, dict)
        or quality_policy.get("interpolation") != "none"
        or quality_policy.get("smoothing") != "none"
    ):
        raise ValueError("39-point quality policy must forbid repair")
    try:
        confidence_threshold = float(
            quality_policy["confidence_threshold"]
        )
        jump_threshold = float(
            quality_policy["temporal_jump_torso_fraction"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("39-point quality thresholds are malformed") from error
    if (
        not math.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
        or not math.isfinite(jump_threshold)
        or jump_threshold <= 0.0
    ):
        raise ValueError("39-point quality thresholds are invalid")
    if require_source_lineage:
        lineage = trajectory.get("source_lineage")
        if not isinstance(lineage, dict):
            raise ValueError("39-point source_lineage is required")
        for field in (
            "video_sha256",
            "predictions_sha256",
            "corrected_trajectory_sha256",
        ):
            value = lineage.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"39-point source_lineage {field} is invalid")

    invalid_by_role = {
        role: [] for role in SUPERANIMAL_QUADRUPED_39
    }
    flags_by_role = {
        role: Counter() for role in SUPERANIMAL_QUADRUPED_39
    }
    identity_frames = {
        "front": {"swapped_frames": [], "ambiguous_frames": []},
        "rear": {"swapped_frames": [], "ambiguous_frames": []},
    }
    previous_time = -math.inf
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_idx") != frame_idx:
            raise ValueError("39-point frames must be contiguous from zero")
        try:
            timestamp = float(frame["timestamp_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"39-point frame {frame_idx} timestamp is malformed"
            ) from error
        if (
            not math.isfinite(timestamp)
            or timestamp <= previous_time
            or not math.isclose(
                timestamp,
                frame_idx / fps,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"39-point frame {frame_idx} timestamp is invalid"
            )
        previous_time = timestamp
        points = frame.get("keypoints")
        if not isinstance(points, dict) or tuple(points) != SUPERANIMAL_QUADRUPED_39:
            raise ValueError(f"39-point frame {frame_idx} roles differ")
        for role, point in points.items():
            if not isinstance(point, dict) or not isinstance(
                point.get("valid"),
                bool,
            ):
                raise ValueError(f"frame {frame_idx} role {role} validity is malformed")
            flags = point.get("flags")
            if (
                not isinstance(flags, list)
                or any(not isinstance(flag, str) or not flag for flag in flags)
                or len(set(flags)) != len(flags)
            ):
                raise ValueError(f"frame {frame_idx} role {role} flags are malformed")
            confidence = point.get("confidence")
            if not _finite_or_none(confidence) or (
                confidence is not None and not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError(
                    f"frame {frame_idx} role {role} confidence is invalid"
                )
            raw_x = point.get("raw_x_px")
            raw_y = point.get("raw_y_px")
            if not _finite_or_none(raw_x) or not _finite_or_none(raw_y):
                raise ValueError(
                    f"frame {frame_idx} role {role} raw coordinates are invalid"
                )
            if not isinstance(point.get("identity_corrected"), bool):
                raise ValueError(
                    f"frame {frame_idx} role {role} identity flag is malformed"
                )
            if point["valid"]:
                try:
                    x = float(point["x_px"])
                    y = float(point["y_px"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"frame {frame_idx} role {role} valid coordinates are malformed"
                    ) from error
                if (
                    not np.isfinite([x, y]).all()
                    or not 0.0 <= x < width
                    or not 0.0 <= y < height
                    or confidence is None
                    or flags
                ):
                    raise ValueError(
                        f"frame {frame_idx} role {role} valid point is inconsistent"
                    )
            else:
                if point.get("x_px") is not None or point.get("y_px") is not None:
                    raise ValueError(
                        f"frame {frame_idx} role {role} invalid point coordinates "
                        "must be null"
                    )
                if not flags:
                    raise ValueError(
                        f"frame {frame_idx} role {role} invalid point needs flags"
                    )
                invalid_by_role[role].append(frame_idx)
                flags_by_role[role].update(flags)
        for prefix in ("front", "back"):
            report_name = "front" if prefix == "front" else "rear"
            leg_roles = [
                f"{prefix}_{side}_{joint}"
                for side in ("left", "right")
                for joint in _JOINTS
            ]
            ambiguous = [
                "identity_ambiguous" in points[role]["flags"]
                for role in leg_roles
            ]
            if any(ambiguous) and not all(ambiguous):
                raise ValueError(
                    f"frame {frame_idx} {prefix} identity ambiguity "
                    "must invalidate both complete chains"
                )
            if all(ambiguous):
                identity_frames[report_name]["ambiguous_frames"].append(
                    frame_idx
                )
            corrected = [
                points[role]["identity_corrected"]
                for role in leg_roles
            ]
            if any(corrected) and not all(corrected):
                raise ValueError(
                    f"frame {frame_idx} {prefix} identity correction "
                    "must cover both complete chains"
                )
            if all(corrected):
                identity_frames[report_name]["swapped_frames"].append(
                    frame_idx
                )

    if quality_report is not None:
        if quality_report.get("schema") != "qianji.keypoint_39_quality_report":
            raise ValueError("unsupported 39-point quality report schema")
        if quality_report.get("schema_version") != "0.1.0":
            raise ValueError("unsupported 39-point quality report schema_version")
        for field in ("interpolation_applied", "smoothing_applied"):
            if field not in quality_report or quality_report[field] is not False:
                raise ValueError(f"quality report {field} must be explicit false")
        if (
            quality_report.get("frame_count") != frame_count
            or quality_report.get("role_count") != len(SUPERANIMAL_QUADRUPED_39)
        ):
            raise ValueError("39-point quality report counts differ")
        if quality_report.get("identity_anchor") != anchor:
            raise ValueError("39-point quality report identity anchor differs")
        if require_source_lineage and quality_report.get(
            "source_lineage"
        ) != trajectory.get("source_lineage"):
            raise ValueError("39-point quality report source lineage differs")
        identity_report = quality_report.get("identity")
        if (
            not isinstance(identity_report, dict)
            or set(identity_report) != {"front", "rear"}
        ):
            raise ValueError("39-point quality identity report is malformed")
        for report_name, assignment_field in (
            ("front", "front_assignment"),
            ("rear", "rear_assignment"),
        ):
            expected_identity = {
                "anchor_assignment": anchor[assignment_field],
                **identity_frames[report_name],
            }
            if identity_report.get(report_name) != expected_identity:
                raise ValueError(
                    f"39-point quality {report_name} identity report differs"
                )
        role_reports = quality_report.get("roles")
        if not isinstance(role_reports, dict) or tuple(role_reports) != (
            SUPERANIMAL_QUADRUPED_39
        ):
            raise ValueError("39-point role quality reports differ")
        for role in SUPERANIMAL_QUADRUPED_39:
            item = role_reports[role]
            if (
                not isinstance(item, dict)
                or item.get("invalid_frames") != invalid_by_role[role]
                or item.get("valid_frames")
                != frame_count - len(invalid_by_role[role])
                or item.get("flag_counts")
                != dict(sorted(flags_by_role[role].items()))
            ):
                raise ValueError(f"quality report role {role!r} disagrees")
    return {
        "frames": frame_count,
        "roles": len(SUPERANIMAL_QUADRUPED_39),
        "invalid_points": sum(map(len, invalid_by_role.values())),
    }


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
    if not dataframe.index.equals(pd.RangeIndex(video.frame_count)):
        raise ValueError(
            "H5 frame index must be the exact zero-based contiguous range"
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
    if values.shape != (len(dataframe), 3):
        raise ValueError(f"bodypart {role!r} has malformed x/y/likelihood")
    finite_likelihood = np.isfinite(values[:, 2])
    if (
        finite_likelihood
        & ((values[:, 2] < 0.0) | (values[:, 2] > 1.0))
    ).any():
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
    ambiguous_frames = {
        report_name: set(section["ambiguous_frames"])
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
            finite_xy = bool(np.isfinite([x, y]).all())
            finite_confidence = math.isfinite(float(confidence))
            if not finite_xy or not finite_confidence:
                flags.append("non_finite")
            if finite_confidence and confidence < confidence_threshold:
                flags.append("low_confidence")
            if finite_xy and (
                not 0.0 <= x < video.width
                or not 0.0 <= y < video.height
            ):
                flags.append("out_of_bounds")
            if frame_idx > 0 and finite_xy and np.isfinite(
                values[frame_idx - 1, :2]
            ).all():
                step = float(
                    np.linalg.norm(
                        values[frame_idx, :2]
                        - values[frame_idx - 1, :2]
                    )
                )
                if step > 0.75 * torso_scales[frame_idx]:
                    flags.append("temporal_jump")
            if (
                prefix is not None
                and frame_idx in ambiguous_frames[prefix]
            ):
                flags.append("identity_ambiguous")
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
                    "raw_x_px": float(x) if math.isfinite(float(x)) else None,
                    "raw_y_px": float(y) if math.isfinite(float(y)) else None,
                    "confidence": (
                        float(confidence) if finite_confidence else None
                    ),
                    "valid": valid,
                    "identity_corrected": identity_corrected,
                    "flags": flags,
                }
            )
        finite_confidences = values[
            np.isfinite(values[:, 2]),
            2,
        ]
        confidence_summary = {
            "minimum": (
                float(np.min(finite_confidences))
                if len(finite_confidences)
                else None
            ),
            "median": (
                float(np.median(finite_confidences))
                if len(finite_confidences)
                else None
            ),
            "maximum": (
                float(np.max(finite_confidences))
                if len(finite_confidences)
                else None
            ),
        }
        role_reports[role] = {
            "invalid_frames": invalid_frames,
            "valid_frames": len(values) - len(invalid_frames),
            "flag_counts": dict(sorted(flags_counter.items())),
            "confidence": confidence_summary,
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
        "identity_anchor": {
            "frame_idx": anchor_frame,
            "front_assignment": front_anchor,
            "rear_assignment": rear_anchor,
        },
        "torso_scale_fallback_frames": torso_fallback_frames,
        "identity": identity_report,
        "roles": role_reports,
        "interpolation_applied": False,
        "smoothing_applied": False,
    }
    return Observation39Result(trajectory=trajectory, report=report)
