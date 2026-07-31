"""Validation and packaging primitives for QianJi VGT model sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qianji_animal_motion.lift_3d import KEYPOINT_ROLES


REACHABILITY_THRESHOLDS = {
    "feasible_keypoint_error_m": 0.02,
    "feasible_edge_violation_m": 0.0001,
    "feasible_clipped_fraction": 0.05,
    "marginal_keypoint_error_m": 0.05,
    "marginal_clipped_fraction": 0.15,
}


@dataclass(frozen=True)
class VgtSequence:
    site_names: tuple[str, ...]
    times: np.ndarray
    positions: np.ndarray
    rods: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ControlMotion:
    times: np.ndarray
    positions: np.ndarray
    confidences: np.ndarray


def validate_control_motion(
    motion: dict,
    *,
    expected_frames: int,
    expected_fps: float,
) -> ControlMotion:
    """Validate one exact six-role QianJi control-motion trajectory."""
    if motion.get("schema") != "qianji-keypoint-trajectory-v1":
        raise ValueError("control motion schema is unsupported")
    try:
        fps = float(motion["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("control motion fps must be finite") from error
    if (
        not math.isfinite(fps)
        or not math.isfinite(expected_fps)
        or expected_fps <= 0.0
        or not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("control motion fps does not match expected fps")
    frames = motion.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise ValueError(
            f"control motion must contain exactly {expected_frames} frames"
        )
    times = np.empty(expected_frames, dtype=float)
    positions = np.empty((expected_frames, len(KEYPOINT_ROLES), 3), dtype=float)
    confidences = np.empty((expected_frames, len(KEYPOINT_ROLES)), dtype=float)
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"control frame {frame_idx} must be an object")
        points = frame.get("keypoints")
        if (
            not isinstance(points, dict)
            or set(points) != set(KEYPOINT_ROLES)
        ):
            raise ValueError(
                f"control frame {frame_idx} must contain the exact six control roles"
            )
        try:
            time = float(frame["time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"control frame {frame_idx} time must be finite"
            ) from error
        if not math.isfinite(time):
            raise ValueError(f"control frame {frame_idx} time must be finite")
        times[frame_idx] = time
        for role_idx, role in enumerate(KEYPOINT_ROLES):
            try:
                value = np.asarray(points[role], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"control frame {frame_idx} role {role!r} must contain XYZC"
                ) from error
            if value.shape != (4,) or not np.isfinite(value).all():
                raise ValueError(
                    f"control frame {frame_idx} role {role!r} must contain finite XYZC"
                )
            if not 0.0 <= value[3] <= 1.0:
                raise ValueError(
                    f"control frame {frame_idx} role {role!r} confidence "
                    "must be within [0, 1]"
                )
            positions[frame_idx, role_idx] = value[:3]
            confidences[frame_idx, role_idx] = value[3]
    if expected_frames > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("control motion times must be strictly increasing")
    expected_times = np.arange(expected_frames, dtype=float) / expected_fps
    if not np.allclose(times, expected_times, rtol=0.0, atol=1e-6):
        raise ValueError("control motion times must match frame_idx/fps")
    for array in (times, positions, confidences):
        array.setflags(write=False)
    return ControlMotion(
        times=times,
        positions=positions,
        confidences=confidences,
    )


def validate_control_pair_against_sequence(
    desired: dict,
    projected: dict,
    sequence: VgtSequence,
    rig: dict,
    *,
    expected_frames: int,
    expected_fps: float,
    position_tolerance_m: float = 1e-9,
) -> tuple[ControlMotion, ControlMotion]:
    """Cross-check desired/projected controls and projected VGT site targets."""
    desired_motion = validate_control_motion(
        desired,
        expected_frames=expected_frames,
        expected_fps=expected_fps,
    )
    projected_motion = validate_control_motion(
        projected,
        expected_frames=expected_frames,
        expected_fps=expected_fps,
    )
    if not np.allclose(
        desired_motion.times,
        projected_motion.times,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("desired and projected control times differ")
    if not np.allclose(
        projected_motion.times,
        sequence.times,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("projected control and VGT sequence times differ")
    if not math.isfinite(position_tolerance_m) or position_tolerance_m < 0.0:
        raise ValueError("position tolerance must be finite and non-negative")
    site_map = rig.get("key_site_map")
    if (
        rig.get("schema") != "qianji-key-site-map"
        or not isinstance(site_map, dict)
        or set(site_map) != set(KEYPOINT_ROLES)
    ):
        raise ValueError("rig must map the exact six control roles")
    site_indices = {name: index for index, name in enumerate(sequence.site_names)}
    for role_idx, role in enumerate(KEYPOINT_ROLES):
        site_name = site_map[role]
        if site_name not in site_indices:
            raise ValueError(f"control role {role!r} references a missing VGT site")
        expected = sequence.positions[:, site_indices[site_name], :]
        actual = projected_motion.positions[:, role_idx, :]
        if not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=position_tolerance_m,
        ):
            raise ValueError(
                f"projected control motion for {role!r} does not match VGT NPZ"
            )
    return desired_motion, projected_motion


def compute_rod_constraint_metrics(
    sequence: VgtSequence,
    robot: dict,
    *,
    violation_epsilon_m: float = 1e-12,
) -> dict:
    """Recompute every rod's permitted node-length violation from the NPZ."""
    try:
        port_offset = float(robot.get("port_offset", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("robot port_offset must be finite") from error
    if (
        not math.isfinite(port_offset)
        or port_offset < 0.0
        or not math.isfinite(violation_epsilon_m)
        or violation_epsilon_m < 0.0
    ):
        raise ValueError("rod metric tolerances must be finite and non-negative")
    rods = robot.get("rod_groups")
    if not isinstance(rods, list) or len(rods) != len(sequence.rods):
        raise ValueError("robot rods do not match the VGT sequence topology")
    indices = {name: index for index, name in enumerate(sequence.site_names)}
    violations = np.empty((len(sequence.times), len(rods)), dtype=float)
    for rod_idx, rod in enumerate(rods):
        if not isinstance(rod, dict):
            raise ValueError(f"rod {rod_idx} must be an object")
        site1 = rod.get("site1")
        site2 = rod.get("site2")
        if site1 not in indices or site2 not in indices or site1 == site2:
            raise ValueError(f"rod {rod_idx} references invalid endpoints")
        constraint = rod.get("constraint")
        if not isinstance(constraint, dict):
            raise ValueError(f"rod {rod_idx} constraint must be an object")
        try:
            minimum = float(constraint["effective_min_length"])
            maximum = float(constraint["effective_max_length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"rod {rod_idx} effective limits are malformed"
            ) from error
        if (
            not np.isfinite([minimum, maximum]).all()
            or minimum < 0.0
            or maximum < minimum
        ):
            raise ValueError(f"rod {rod_idx} effective limits are invalid")
        lengths = np.linalg.norm(
            sequence.positions[:, indices[site2], :]
            - sequence.positions[:, indices[site1], :],
            axis=1,
        )
        minimum_node = minimum + 2.0 * port_offset
        maximum_node = maximum + 2.0 * port_offset
        violations[:, rod_idx] = np.maximum.reduce(
            (
                minimum_node - lengths,
                lengths - maximum_node,
                np.zeros_like(lengths),
            )
        )
    frame_max = np.max(violations, axis=1)
    frame_fraction = np.mean(
        violations > violation_epsilon_m,
        axis=1,
    )
    return {
        "max_edge_violation_m": float(np.max(frame_max)),
        "max_violated_rod_fraction": float(np.max(frame_fraction)),
        "mean_edge_violation_m": float(np.mean(violations)),
        "frame_max_edge_violation_m": frame_max.astype(float).tolist(),
        "frame_violated_rod_fraction": frame_fraction.astype(float).tolist(),
    }


def recompute_reachability_metrics(
    desired: ControlMotion,
    projected: ControlMotion,
    sequence: VgtSequence,
    robot: dict,
) -> dict:
    """Independently derive QianJi control errors, geometry, and frame status."""
    if (
        desired.positions.shape != projected.positions.shape
        or desired.positions.shape[:2]
        != (len(sequence.times), len(KEYPOINT_ROLES))
    ):
        raise ValueError("control motions do not match the VGT sequence")
    errors = np.linalg.norm(projected.positions - desired.positions, axis=2)
    geometry = compute_rod_constraint_metrics(
        sequence,
        robot,
        violation_epsilon_m=1e-8,
    )
    frame_edge = np.asarray(
        geometry["frame_max_edge_violation_m"],
        dtype=float,
    )
    frame_clipped = np.asarray(
        geometry["frame_violated_rod_fraction"],
        dtype=float,
    )
    records = []
    for frame_idx in range(len(sequence.times)):
        max_error = float(np.max(errors[frame_idx]))
        mean_error = float(np.mean(errors[frame_idx]))
        edge_error = float(frame_edge[frame_idx])
        clipped_fraction = float(frame_clipped[frame_idx])
        if (
            max_error
            <= REACHABILITY_THRESHOLDS["feasible_keypoint_error_m"]
            and edge_error
            <= REACHABILITY_THRESHOLDS["feasible_edge_violation_m"]
            and clipped_fraction
            <= REACHABILITY_THRESHOLDS["feasible_clipped_fraction"]
        ):
            status = "feasible"
        elif (
            max_error
            <= REACHABILITY_THRESHOLDS["marginal_keypoint_error_m"]
            or clipped_fraction
            <= REACHABILITY_THRESHOLDS["marginal_clipped_fraction"]
        ):
            status = "marginal"
        else:
            status = "unreachable"
        records.append(
            {
                "frame": frame_idx,
                "time": float(sequence.times[frame_idx]),
                "status": status,
                "max_keypoint_error_m": max_error,
                "mean_keypoint_error_m": mean_error,
                "keypoint_errors_m": {
                    role: float(errors[frame_idx, role_idx])
                    for role_idx, role in enumerate(KEYPOINT_ROLES)
                },
                "max_edge_violation_m": edge_error,
                "estimated_clipped_fraction": clipped_fraction,
            }
        )
    status_counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("feasible", "marginal", "unreachable")
    }
    count = len(records)
    summary = {
        "frames": count,
        "status_counts": status_counts,
        "feasible_fraction": status_counts["feasible"] / count,
        "marginal_or_feasible_fraction": (
            status_counts["feasible"] + status_counts["marginal"]
        )
        / count,
        "max_keypoint_error_m": max(
            record["max_keypoint_error_m"] for record in records
        ),
        "mean_keypoint_error_m": float(
            np.mean(
                [record["mean_keypoint_error_m"] for record in records]
            )
        ),
        "max_edge_violation_m": max(
            record["max_edge_violation_m"] for record in records
        ),
        "max_estimated_clipped_fraction": max(
            record["estimated_clipped_fraction"] for record in records
        ),
        "mean_estimated_clipped_fraction": float(
            np.mean(
                [
                    record["estimated_clipped_fraction"]
                    for record in records
                ]
            )
        ),
    }
    return {
        "thresholds": dict(REACHABILITY_THRESHOLDS),
        "summary": summary,
        "frames": records,
        "geometry": geometry,
    }


def _same_number(actual: object, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not math.isfinite(value) or not math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label} differs from independently recomputed value")


def validate_reachability_report(
    report: dict,
    recomputed: dict,
    desired: ControlMotion,
    projected: ControlMotion,
) -> None:
    """Reject a QianJi report that disagrees with independent artifacts."""
    if report.get("schema") != "qianji-keypoint-reachability-report-v1":
        raise ValueError("reachability report schema is unsupported")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(
        REACHABILITY_THRESHOLDS
    ):
        raise ValueError("reachability thresholds are missing or changed")
    for name, expected in REACHABILITY_THRESHOLDS.items():
        _same_number(thresholds[name], expected, f"reachability threshold {name}")

    summary = report.get("summary")
    expected_summary = recomputed["summary"]
    if not isinstance(summary, dict) or set(summary) != set(expected_summary):
        raise ValueError("reachability summary contract differs")
    if summary.get("frames") != expected_summary["frames"]:
        raise ValueError("reachability summary frame count differs")
    if summary.get("status_counts") != expected_summary["status_counts"]:
        raise ValueError("reachability status counts differ")
    for name in set(expected_summary) - {"frames", "status_counts"}:
        _same_number(
            summary[name],
            expected_summary[name],
            f"reachability summary {name}",
        )

    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) != len(
        recomputed["frames"]
    ):
        raise ValueError("reachability frame reports differ")
    for frame_idx, (actual, expected) in enumerate(
        zip(frames, recomputed["frames"], strict=True)
    ):
        if (
            not isinstance(actual, dict)
            or actual.get("frame") != frame_idx
            or actual.get("status") != expected["status"]
        ):
            raise ValueError(f"reachability frame {frame_idx} status differs")
        for name in (
            "time",
            "max_keypoint_error_m",
            "mean_keypoint_error_m",
            "max_edge_violation_m",
            "estimated_clipped_fraction",
        ):
            _same_number(
                actual.get(name),
                expected[name],
                f"reachability frame {frame_idx} {name}",
            )
        keypoint_errors = actual.get("keypoint_errors_m")
        if (
            not isinstance(keypoint_errors, dict)
            or set(keypoint_errors) != set(KEYPOINT_ROLES)
        ):
            raise ValueError(
                f"reachability frame {frame_idx} keypoint errors differ"
            )
        for role in KEYPOINT_ROLES:
            _same_number(
                keypoint_errors[role],
                expected["keypoint_errors_m"][role],
                f"reachability frame {frame_idx} role {role} error",
            )
        for label, motion in (
            ("target_keypoints", desired),
            ("projected_keypoints", projected),
        ):
            points = actual.get(label)
            if not isinstance(points, dict) or set(points) != set(KEYPOINT_ROLES):
                raise ValueError(
                    f"reachability frame {frame_idx} {label} roles differ"
                )
            for role_idx, role in enumerate(KEYPOINT_ROLES):
                value = np.asarray(points[role], dtype=float)
                expected_value = np.concatenate(
                    (
                        motion.positions[frame_idx, role_idx],
                        [motion.confidences[frame_idx, role_idx]],
                    )
                )
                compare_value = (
                    value[:3] if label == "target_keypoints" else value
                )
                compare_expected = (
                    expected_value[:3]
                    if label == "target_keypoints"
                    else expected_value
                )
                if (
                    value.shape != (4,)
                    or not np.isfinite(value).all()
                    or not 0.0 <= value[3] <= 1.0
                    or not np.allclose(
                        compare_value,
                        compare_expected,
                        rtol=0.0,
                        atol=1e-9,
                    )
                ):
                    raise ValueError(
                        f"reachability frame {frame_idx} {label} differs"
                    )


def _validate_rods(
    robot: dict,
    site_names: tuple[str, ...],
    *,
    expected_rods: int,
) -> tuple[tuple[str, str], ...]:
    rods = robot.get("rod_groups")
    if not isinstance(rods, list) or len(rods) != expected_rods:
        raise ValueError(f"robot must contain exactly {expected_rods} rods")
    site_set = set(site_names)
    output = []
    rod_names = set()
    for index, rod in enumerate(rods):
        if not isinstance(rod, dict):
            raise ValueError(f"rod {index} must be an object")
        name = rod.get("name")
        site1 = rod.get("site1")
        site2 = rod.get("site2")
        if not isinstance(name, str) or not name or name in rod_names:
            raise ValueError("rod names must be non-empty and unique")
        rod_names.add(name)
        if site1 not in site_set or site2 not in site_set:
            raise ValueError(f"rod {name!r} references a missing site")
        if site1 == site2:
            raise ValueError(f"rod {name!r} must connect different sites")
        output.append((site1, site2))
    return tuple(output)


def load_and_validate_vgt_sequence(
    npz_path: Path,
    robot: dict,
    *,
    expected_frames: int,
    expected_sites: int,
    expected_rods: int,
) -> VgtSequence:
    """Load finite QianJi site targets with an exact robot topology contract."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"VGT sequence does not exist: {path}")
    if expected_frames <= 0 or expected_sites <= 0 or expected_rods <= 0:
        raise ValueError("expected counts must be positive")
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = {"site_names", "times", "positions"} - set(archive.files)
            if missing:
                raise ValueError(
                    "VGT NPZ is missing arrays: " + ", ".join(sorted(missing))
                )
            raw_names = np.array(archive["site_names"], copy=True)
            times = np.asarray(archive["times"], dtype=float).copy()
            positions = np.asarray(archive["positions"], dtype=float).copy()
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("VGT NPZ"):
            raise
        raise ValueError(f"cannot load VGT NPZ: {path}") from error

    if raw_names.shape != (expected_sites,):
        raise ValueError(
            f"site_names must contain exactly {expected_sites} sites"
        )
    if raw_names.dtype.kind not in {"U", "S"}:
        raise ValueError("site_names must be a string array")
    site_names = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_names.tolist()
    )
    if any(not name for name in site_names) or len(set(site_names)) != len(
        site_names
    ):
        raise ValueError("site_names must be non-empty and unique")

    sites = robot.get("sites")
    if not isinstance(sites, dict) or len(sites) != expected_sites:
        raise ValueError(f"robot must contain exactly {expected_sites} sites")
    if site_names != tuple(sites):
        raise ValueError("site_names must exactly match robot site order")
    if times.shape != (expected_frames,):
        raise ValueError(f"times must contain exactly {expected_frames} frames")
    if positions.shape != (expected_frames, expected_sites, 3):
        raise ValueError(
            "positions shape must be "
            f"({expected_frames}, {expected_sites}, 3) frames/sites/XYZ"
        )
    if not np.isfinite(times).all() or not np.isfinite(positions).all():
        raise ValueError("VGT times and positions must be finite")
    if expected_frames > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("VGT times must be strictly increasing")
    rods = _validate_rods(
        robot,
        site_names,
        expected_rods=expected_rods,
    )
    times.setflags(write=False)
    positions.setflags(write=False)
    return VgtSequence(
        site_names=site_names,
        times=times,
        positions=positions,
        rods=rods,
    )


def validate_render_rate(sequence: VgtSequence, fps: float) -> None:
    """Validate the declared display rate without resampling the sequence."""
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("render fps must be finite and positive")
    if sequence.times.size > 1:
        expected_step = 1.0 / fps
        actual_steps = np.diff(sequence.times)
        if not np.allclose(actual_steps, expected_step, atol=1e-6, rtol=1e-5):
            raise ValueError("VGT times do not match the requested render fps")
