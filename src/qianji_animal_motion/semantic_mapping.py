"""Convert DeepLabCut quadruped predictions into six semantic keypoints."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEMANTIC_KEYPOINTS = (
    "spine_front",
    "spine_rear",
    "front_left_foot",
    "front_right_foot",
    "rear_left_foot",
    "rear_right_foot",
)

REQUIRED_BODYPARTS = (
    "back_base",
    "back_middle",
    "back_end",
    "neck_end",
    "tail_base",
    "front_left_thai",
    "front_left_knee",
    "front_left_paw",
    "front_right_thai",
    "front_right_knee",
    "front_right_paw",
    "back_left_thai",
    "back_left_knee",
    "back_left_paw",
    "back_right_thai",
    "back_right_knee",
    "back_right_paw",
)

_JOINT_WEIGHTS = np.asarray([0.2, 0.3, 0.5], dtype=float)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class IdentityResolution:
    states: np.ndarray
    ambiguous: np.ndarray


@dataclass(frozen=True)
class MappingResult:
    trajectory: dict
    report: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dataframe(df: pd.DataFrame, individual: str) -> str:
    expected_levels = {"scorer", "individuals", "bodyparts", "coords"}
    if set(df.columns.names) != expected_levels:
        raise ValueError(
            "DeepLabCut columns must contain scorer, individuals, bodyparts, and coords"
        )
    scorers = list(dict.fromkeys(df.columns.get_level_values("scorer")))
    individuals = set(df.columns.get_level_values("individuals"))
    bodyparts = set(df.columns.get_level_values("bodyparts"))
    coords = set(df.columns.get_level_values("coords"))
    missing = sorted(set(REQUIRED_BODYPARTS) - bodyparts)
    if len(scorers) != 1:
        raise ValueError(f"expected one scorer, found {scorers}")
    if individual not in individuals:
        raise ValueError(f"individual {individual!r} not found")
    if missing:
        raise ValueError("missing required bodyparts: " + ", ".join(missing))
    if not {"x", "y", "likelihood"}.issubset(coords):
        raise ValueError("predictions must contain x, y, and likelihood coordinates")
    return str(scorers[0])


def _bodypart_array(
    df: pd.DataFrame,
    scorer: str,
    individual: str,
    bodypart: str,
) -> np.ndarray:
    return df.loc[
        :,
        pd.IndexSlice[scorer, individual, bodypart, ["x", "y", "likelihood"]],
    ].to_numpy(dtype=float)


def _leg_chains(
    df: pd.DataFrame,
    scorer: str,
    individual: str,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.stack(
        [
            _bodypart_array(df, scorer, individual, f"{prefix}_left_{joint}")
            for joint in ("thai", "knee", "paw")
        ],
        axis=1,
    )
    right = np.stack(
        [
            _bodypart_array(df, scorer, individual, f"{prefix}_right_{joint}")
            for joint in ("thai", "knee", "paw")
        ],
        axis=1,
    )
    return left, right


def _ordered_pair(
    left: np.ndarray,
    right: np.ndarray,
    frame: int,
    state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if state == 0:
        return left[frame], right[frame]
    return right[frame], left[frame]


def _transition_cost_between(
    left: np.ndarray,
    right: np.ndarray,
    previous_frame: int,
    frame: int,
    previous_state: int,
    state: int,
    scale: float,
    switch_penalty: float,
) -> float:
    previous = _ordered_pair(left, right, previous_frame, previous_state)
    current = _ordered_pair(left, right, frame, state)
    cost = 0.0
    usable = 0
    gap_scale = scale * math.sqrt(max(abs(frame - previous_frame), 1))
    for previous_leg, current_leg in zip(previous, current, strict=True):
        confidence = np.minimum(previous_leg[:, 2], current_leg[:, 2])
        finite = np.isfinite(previous_leg[:, :2]).all(axis=1) & np.isfinite(
            current_leg[:, :2]
        ).all(axis=1)
        reliable = finite & (confidence >= 0.1)
        if not reliable.any():
            continue
        distance = np.linalg.norm(
            current_leg[reliable, :2] - previous_leg[reliable, :2], axis=1
        )
        weights = _JOINT_WEIGHTS[reliable] * confidence[reliable]
        cost += float(np.sum(weights * np.square(distance / gap_scale)))
        usable += int(reliable.sum())
    if usable == 0:
        return math.inf
    if state != previous_state:
        cost += switch_penalty
    return cost


def resolve_leg_identities(
    left: np.ndarray,
    right: np.ndarray,
    *,
    scale: float,
    anchor_frame: int = 0,
    anchor_state: int = 0,
    switch_penalty: float = 0.0025,
    ambiguity_margin: float = 0.15,
) -> IdentityResolution:
    """Resolve keep/swap states in both directions from a confirmed anchor."""
    frame_count = len(left)
    if frame_count != len(right):
        raise ValueError("left and right leg sequences must have equal length")
    if frame_count == 0:
        return IdentityResolution(
            states=np.empty(0, dtype=np.int8),
            ambiguous=np.empty(0, dtype=bool),
        )
    if not 0 <= anchor_frame < frame_count:
        raise ValueError("anchor frame must be within the leg sequence")
    if anchor_state not in (0, 1):
        raise ValueError("anchor state must be 0 (keep) or 1 (swap)")

    scale = max(float(scale), 1.0)
    states = np.full(frame_count, anchor_state, dtype=np.int8)
    ambiguous = np.zeros(frame_count, dtype=bool)

    def track(indices: range) -> None:
        last_reliable_frame = anchor_frame
        last_reliable_state = anchor_state
        for frame in indices:
            alternatives = np.asarray(
                [
                    _transition_cost_between(
                        left,
                        right,
                        last_reliable_frame,
                        frame,
                        last_reliable_state,
                        state,
                        scale,
                        switch_penalty,
                    )
                    for state in (0, 1)
                ],
                dtype=float,
            )
            if not np.isfinite(alternatives).all():
                states[frame] = last_reliable_state
                ambiguous[frame] = True
                continue

            chosen = int(np.argmin(alternatives))
            denominator = max(float(np.max(alternatives)), 1e-12)
            margin = abs(float(alternatives[0] - alternatives[1])) / denominator
            if margin < ambiguity_margin:
                states[frame] = last_reliable_state
                ambiguous[frame] = True
                last_reliable_frame = frame
                continue

            states[frame] = chosen
            last_reliable_frame = frame
            last_reliable_state = chosen

    track(range(anchor_frame + 1, frame_count))
    track(range(anchor_frame - 1, -1, -1))
    return IdentityResolution(states=states, ambiguous=ambiguous)


def _candidate_point(
    x: float,
    y: float,
    confidence: float,
    *,
    source: str | list[str],
    width: int,
    height: int,
    confidence_threshold: float,
    identity_corrected: bool = False,
    identity_ambiguous: bool = False,
    fallback_used: bool = False,
) -> dict:
    flags: list[str] = []
    if not np.isfinite([x, y, confidence]).all():
        flags.append("non_finite")
    if confidence < confidence_threshold:
        flags.append("low_confidence")
    if np.isfinite([x, y]).all() and not (0 <= x < width and 0 <= y < height):
        flags.append("out_of_bounds")
    if identity_ambiguous:
        flags.append("identity_ambiguous")
    valid = not flags
    output_confidence = (
        float(confidence) if np.isfinite(confidence) else None
    )
    return {
        "x_px": float(x) if valid else None,
        "y_px": float(y) if valid else None,
        "confidence": output_confidence,
        "valid": valid,
        "source": source,
        "identity_corrected": identity_corrected,
        "fallback_used": fallback_used,
        "flags": flags,
        "_candidate_xy": (float(x), float(y)),
    }


def _calibrate_fallback_offset(
    primary: np.ndarray,
    fallback: np.ndarray,
    *,
    confidence_threshold: float,
    torso_scale: float,
) -> tuple[np.ndarray | None, dict]:
    """Learn a robust same-frame fallback offset from mutually reliable frames."""
    finite = np.isfinite(primary).all(axis=1) & np.isfinite(fallback).all(axis=1)
    reliable = (
        finite
        & (primary[:, 2] >= confidence_threshold)
        & (fallback[:, 2] >= confidence_threshold)
    )
    sample_count = int(reliable.sum())
    report = {
        "calibration_frames": sample_count,
        "validated": False,
        "median_offset_px": None,
        "residual_p90_px": None,
    }
    if sample_count < 3:
        return None, report

    offsets = primary[reliable, :2] - fallback[reliable, :2]
    median_offset = np.median(offsets, axis=0)
    residuals = np.linalg.norm(offsets - median_offset, axis=1)
    residual_p90 = float(np.quantile(residuals, 0.9))
    stable = residual_p90 <= max(5.0, 0.08 * torso_scale)
    report.update(
        {
            "validated": stable,
            "median_offset_px": median_offset.astype(float).tolist(),
            "residual_p90_px": residual_p90,
        }
    )
    return (median_offset if stable else None), report


def _spine_point(
    primary: np.ndarray,
    fallback: np.ndarray,
    frame_index: int,
    *,
    primary_name: str,
    fallback_name: str,
    fallback_offset: np.ndarray | None,
    width: int,
    height: int,
    confidence_threshold: float,
) -> dict:
    primary_point = primary[frame_index]
    candidate = _candidate_point(
        primary_point[0],
        primary_point[1],
        primary_point[2],
        source=primary_name,
        width=width,
        height=height,
        confidence_threshold=confidence_threshold,
    )
    if candidate["valid"]:
        return candidate

    fallback_point = fallback[frame_index]
    if fallback_offset is not None:
        fallback_candidate = _candidate_point(
            fallback_point[0] + fallback_offset[0],
            fallback_point[1] + fallback_offset[1],
            fallback_point[2],
            source=fallback_name,
            width=width,
            height=height,
            confidence_threshold=confidence_threshold,
            fallback_used=True,
        )
        if fallback_candidate["valid"]:
            return fallback_candidate

    candidate["flags"].append("fallback_unavailable")
    return candidate


def _calibrate_back_extrapolation(
    back_base: np.ndarray,
    back_middle: np.ndarray,
    back_end: np.ndarray,
    *,
    confidence_threshold: float,
    torso_scale: float,
) -> tuple[float | None, dict]:
    finite = (
        np.isfinite(back_base).all(axis=1)
        & np.isfinite(back_middle).all(axis=1)
        & np.isfinite(back_end).all(axis=1)
    )
    reliable = finite & (
        np.minimum.reduce(
            [back_base[:, 2], back_middle[:, 2], back_end[:, 2]]
        )
        >= confidence_threshold
    )
    axis = back_middle[:, :2] - back_end[:, :2]
    axis_squared = np.sum(np.square(axis), axis=1)
    reliable &= axis_squared > 1.0
    sample_count = int(reliable.sum())
    report = {
        "calibration_frames": sample_count,
        "validated": False,
        "median_extrapolation_factor": None,
        "factor_p90_deviation": None,
        "perpendicular_residual_p90_px": None,
    }
    if sample_count < 3:
        return None, report

    delta = back_base[:, :2] - back_end[:, :2]
    factors = np.sum(delta * axis, axis=1)[reliable] / axis_squared[reliable]
    median_factor = float(np.median(factors))
    estimates = (
        back_end[reliable, :2]
        + median_factor * axis[reliable]
    )
    residuals = np.linalg.norm(back_base[reliable, :2] - estimates, axis=1)
    factor_p90 = float(np.quantile(np.abs(factors - median_factor), 0.9))
    residual_p90 = float(np.quantile(residuals, 0.9))
    stable = (
        1.0 < median_factor < 3.0
        and factor_p90 <= 0.25
        and residual_p90 <= max(5.0, 0.12 * torso_scale)
    )
    report.update(
        {
            "validated": stable,
            "median_extrapolation_factor": median_factor,
            "factor_p90_deviation": factor_p90,
            "perpendicular_residual_p90_px": residual_p90,
        }
    )
    return (median_factor if stable else None), report


def _repair_front_spine_from_back(
    point: dict,
    back_end: np.ndarray,
    back_middle: np.ndarray,
    frame_index: int,
    *,
    extrapolation_factor: float | None,
    width: int,
    height: int,
    confidence_threshold: float,
) -> dict:
    if point["valid"] or extrapolation_factor is None:
        return point
    rear = back_end[frame_index]
    middle = back_middle[frame_index]
    xy = rear[:2] + extrapolation_factor * (middle[:2] - rear[:2])
    repaired = _candidate_point(
        xy[0],
        xy[1],
        min(rear[2], middle[2]),
        source=["back_end", "back_middle"],
        width=width,
        height=height,
        confidence_threshold=confidence_threshold,
        fallback_used=True,
    )
    return repaired if repaired["valid"] else point


def _invalidate(point: dict, flag: str) -> None:
    if flag not in point["flags"]:
        point["flags"].append(flag)
    point["valid"] = False
    point["x_px"] = None
    point["y_px"] = None


def _check_back_structure(
    front: dict,
    rear: dict,
    back_middle: np.ndarray,
    *,
    torso_scale: float,
    confidence_threshold: float,
) -> None:
    """Reject dorsal anchors that disagree with the same-frame back polyline."""
    if not front["valid"] or not rear["valid"]:
        return
    if not np.isfinite(back_middle).all() or back_middle[2] < confidence_threshold:
        return

    rear_xy = np.asarray(rear["_candidate_xy"], dtype=float)
    front_xy = np.asarray(front["_candidate_xy"], dtype=float)
    middle_xy = back_middle[:2]
    axis = front_xy - rear_xy
    length = float(np.linalg.norm(axis))
    if length < 0.35 * torso_scale or length > 1.8 * torso_scale:
        _invalidate(front, "spine_distance_outlier")
        _invalidate(rear, "spine_distance_outlier")
        return

    length_squared = float(np.dot(axis, axis))
    progress = float(np.dot(middle_xy - rear_xy, axis) / length_squared)
    projection = rear_xy + progress * axis
    deviation = float(np.linalg.norm(middle_xy - projection))
    if not (-0.2 <= progress <= 1.2) or deviation > 0.35 * length:
        _invalidate(front, "back_polyline_inconsistent")
        _invalidate(rear, "back_polyline_inconsistent")


def _check_limb_structure(
    point: dict,
    leg: np.ndarray,
    *,
    confidence_threshold: float,
) -> None:
    """Reject a paw that collapses to the proximal side of a reliable knee."""
    if not point["valid"] or not np.isfinite(leg).all():
        return
    thigh, knee, paw = leg
    if min(thigh[2], knee[2], paw[2]) < confidence_threshold:
        return
    thigh_to_knee = float(np.linalg.norm(knee[:2] - thigh[:2]))
    thigh_to_paw = float(np.linalg.norm(paw[:2] - thigh[:2]))
    if thigh_to_knee > 1.0 and thigh_to_paw < 0.6 * thigh_to_knee:
        _invalidate(point, "limb_structure_inconsistent")


def _mark_jump_outliers(frames: list[dict], torso_scale: float) -> None:
    for name in SEMANTIC_KEYPOINTS:
        candidates = np.asarray(
            [frame["keypoints"][name]["_candidate_xy"] for frame in frames],
            dtype=float,
        )
        if len(candidates) < 2:
            continue
        displacement = np.linalg.norm(np.diff(candidates, axis=0), axis=1) / max(
            torso_scale, 1.0
        )
        finite = displacement[np.isfinite(displacement)]
        if len(finite) == 0:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        threshold = max(0.20, median + 6.0 * mad)
        for frame_index in np.flatnonzero(displacement > threshold) + 1:
            point = frames[int(frame_index)]["keypoints"][name]
            if "jump_outlier" not in point["flags"]:
                point["flags"].append("jump_outlier")
            point["valid"] = False
            point["x_px"] = None
            point["y_px"] = None


def _strip_internal_fields(frames: list[dict]) -> None:
    for frame in frames:
        for point in frame["keypoints"].values():
            point.pop("_candidate_xy", None)


def build_semantic_mapping(
    df: pd.DataFrame,
    video: VideoInfo,
    *,
    individual: str = "animal0",
    confidence_threshold: float = 0.5,
    anchor_frame: int = 0,
    front_anchor_state: int = 0,
    rear_anchor_state: int = 0,
    identity_anchor: dict | None = None,
) -> MappingResult:
    """Build JSON-ready six-point trajectories without modifying DLC predictions."""
    scorer = _validate_dataframe(df, individual)
    if len(df) != video.frame_count:
        raise ValueError(
            f"prediction frames ({len(df)}) do not match video frames ({video.frame_count})"
        )

    front_left, front_right = _leg_chains(df, scorer, individual, "front")
    rear_left, rear_right = _leg_chains(df, scorer, individual, "back")
    back_base = _bodypart_array(df, scorer, individual, "back_base")
    back_middle = _bodypart_array(df, scorer, individual, "back_middle")
    back_end = _bodypart_array(df, scorer, individual, "back_end")
    neck_end = _bodypart_array(df, scorer, individual, "neck_end")
    tail_base = _bodypart_array(df, scorer, individual, "tail_base")
    front_center = (front_left[:, 0, :2] + front_right[:, 0, :2]) / 2.0
    rear_center = (rear_left[:, 0, :2] + rear_right[:, 0, :2]) / 2.0
    torso_lengths = np.linalg.norm(front_center - rear_center, axis=1)
    finite_lengths = torso_lengths[np.isfinite(torso_lengths) & (torso_lengths > 1)]
    torso_scale = (
        float(np.median(finite_lengths))
        if len(finite_lengths)
        else math.hypot(video.width, video.height) / 2.0
    )
    front_identity = resolve_leg_identities(
        front_left,
        front_right,
        scale=torso_scale,
        anchor_frame=anchor_frame,
        anchor_state=front_anchor_state,
    )
    rear_identity = resolve_leg_identities(
        rear_left,
        rear_right,
        scale=torso_scale,
        anchor_frame=anchor_frame,
        anchor_state=rear_anchor_state,
    )
    front_fallback_offset, front_fallback_report = _calibrate_fallback_offset(
        back_base,
        neck_end,
        confidence_threshold=confidence_threshold,
        torso_scale=torso_scale,
    )
    rear_fallback_offset, rear_fallback_report = _calibrate_fallback_offset(
        back_end,
        tail_base,
        confidence_threshold=confidence_threshold,
        torso_scale=torso_scale,
    )
    back_extrapolation_factor, back_extrapolation_report = (
        _calibrate_back_extrapolation(
            back_base,
            back_middle,
            back_end,
            confidence_threshold=confidence_threshold,
            torso_scale=torso_scale,
        )
    )

    frames: list[dict] = []
    for frame_index in range(video.frame_count):
        keypoints: dict[str, dict] = {}
        keypoints["spine_front"] = _spine_point(
            back_base,
            neck_end,
            frame_index,
            primary_name="back_base",
            fallback_name="neck_end",
            fallback_offset=front_fallback_offset,
            width=video.width,
            height=video.height,
            confidence_threshold=confidence_threshold,
        )
        keypoints["spine_front"] = _repair_front_spine_from_back(
            keypoints["spine_front"],
            back_end,
            back_middle,
            frame_index,
            extrapolation_factor=back_extrapolation_factor,
            width=video.width,
            height=video.height,
            confidence_threshold=confidence_threshold,
        )
        keypoints["spine_rear"] = _spine_point(
            back_end,
            tail_base,
            frame_index,
            primary_name="back_end",
            fallback_name="tail_base",
            fallback_offset=rear_fallback_offset,
            width=video.width,
            height=video.height,
            confidence_threshold=confidence_threshold,
        )
        _check_back_structure(
            keypoints["spine_front"],
            keypoints["spine_rear"],
            back_middle[frame_index],
            torso_scale=torso_scale,
            confidence_threshold=confidence_threshold,
        )
        for group, output_prefix, left, right, identity in (
            ("front", "front", front_left, front_right, front_identity),
            ("rear", "rear", rear_left, rear_right, rear_identity),
        ):
            state = int(identity.states[frame_index])
            ordered = _ordered_pair(left, right, frame_index, state)
            source_prefix = "front" if group == "front" else "back"
            sources = (
                (f"{source_prefix}_left_paw", f"{source_prefix}_right_paw")
                if state == 0
                else (f"{source_prefix}_right_paw", f"{source_prefix}_left_paw")
            )
            for side_index, side in enumerate(("left", "right")):
                paw = ordered[side_index][2]
                point = _candidate_point(
                    paw[0],
                    paw[1],
                    paw[2],
                    source=sources[side_index],
                    width=video.width,
                    height=video.height,
                    confidence_threshold=confidence_threshold,
                    identity_corrected=bool(state),
                    identity_ambiguous=bool(identity.ambiguous[frame_index]),
                )
                _check_limb_structure(
                    point,
                    ordered[side_index],
                    confidence_threshold=confidence_threshold,
                )
                keypoints[f"{output_prefix}_{side}_foot"] = point
        frames.append(
            {
                "frame_idx": frame_index,
                "timestamp_s": frame_index / video.fps,
                "keypoints": keypoints,
            }
        )

    _mark_jump_outliers(frames, torso_scale)
    _strip_internal_fields(frames)
    point_report = {}
    for name in SEMANTIC_KEYPOINTS:
        points = [frame["keypoints"][name] for frame in frames]
        flags = Counter(flag for point in points for flag in point["flags"])
        finite_confidences = [
            point["confidence"]
            for point in points
            if point["confidence"] is not None
        ]
        point_report[name] = {
            "valid_frames": sum(point["valid"] for point in points),
            "invalid_frames": [
                index for index, point in enumerate(points) if not point["valid"]
            ],
            "mean_confidence": (
                float(np.mean(finite_confidences))
                if finite_confidences
                else None
            ),
            "fallback_used_frames": [
                index
                for index, point in enumerate(points)
                if point["fallback_used"]
            ],
            "flag_counts": dict(sorted(flags.items())),
        }

    mapping = {
        "spine_front": {
            "primary": "back_base",
            "fallback": "neck_end",
            "structural_fallback": ["back_end", "back_middle"],
            "fallback_policy": "same_frame_validated_relationships",
        },
        "spine_rear": {
            "primary": "back_end",
            "fallback": "tail_base",
            "fallback_policy": "same_frame_validated_offset",
        },
        "front_left_foot": "front_left_paw",
        "front_right_foot": "front_right_paw",
        "rear_left_foot": "back_left_paw",
        "rear_right_foot": "back_right_paw",
    }
    trajectory = {
        "schema": "qianji.keypoint_trajectory_2d",
        "schema_version": "1.2.0",
        "coordinate_system": "image_pixels_top_left_origin_x_right_y_down",
        "video": {
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frame_count": video.frame_count,
        },
        "mapping": mapping,
        "frames": frames,
    }
    report = {
        "schema": "qianji.keypoint_mapping_report",
        "schema_version": "1.2.0",
        "scorer": scorer,
        "individual": individual,
        "confidence_threshold": confidence_threshold,
        "torso_scale_px": torso_scale,
        "spine_fallback_calibration": {
            "spine_front": front_fallback_report,
            "spine_rear": rear_fallback_report,
            "spine_front_back_extrapolation": back_extrapolation_report,
        },
        "identity_corrections": {
            "front": np.flatnonzero(front_identity.states).astype(int).tolist(),
            "rear": np.flatnonzero(rear_identity.states).astype(int).tolist(),
        },
        "identity_ambiguous_frames": {
            "front": np.flatnonzero(front_identity.ambiguous).astype(int).tolist(),
            "rear": np.flatnonzero(rear_identity.ambiguous).astype(int).tolist(),
        },
        "keypoints": point_report,
    }
    if identity_anchor is not None:
        trajectory["identity_anchor"] = dict(identity_anchor)
        report["identity_anchor"] = dict(identity_anchor)
    return MappingResult(trajectory=trajectory, report=report)


def write_json_outputs(
    result: MappingResult,
    *,
    output_dir: Path,
    source_h5: Path,
    source_video: Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    source_h5 = Path(source_h5).resolve()
    source_video = Path(source_video).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = dict(result.trajectory)
    trajectory["source"] = {
        "video_path": str(source_video),
        "predictions_path": str(source_h5),
        "predictions_sha256": _sha256(source_h5),
    }
    report = dict(result.report)
    report["source"] = trajectory["source"]
    trajectory_path = output_dir / "keypoint_trajectory_2d.json"
    report_path = output_dir / "mapping_report.json"
    trajectory_text = (
        json.dumps(
            trajectory,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    report_text = (
        json.dumps(
            report,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    trajectory_path.write_text(trajectory_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    return trajectory_path, report_path
