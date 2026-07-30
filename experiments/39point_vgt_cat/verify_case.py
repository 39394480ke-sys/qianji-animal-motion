#!/usr/bin/env python3
"""Verify every artifact and scientific boundary of a 39-point VGT case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from qianji_animal_motion.keypoints_39 import SUPERANIMAL_QUADRUPED_39
from qianji_animal_motion.lift_3d import KEYPOINT_ROLES
from qianji_animal_motion.vgt_sequence import load_and_validate_vgt_sequence


EXPECTED_FRAMES = 272
EXPECTED_FPS = 30.0
EXPECTED_SITES = 12
EXPECTED_RODS = 30

REQUIRED_FILES = (
    "input_manifest.json",
    "observation/keypoint_trajectory_2d_39.json",
    "observation/keypoint_39_quality_report.json",
    "observation/keypoint_39_preview.mp4",
    "initial_model/robot.json",
    "initial_model/rig_keypoints.json",
    "landmarks/neutral_landmarks_39.json",
    "landmarks/keypoint_motion_3d_39.json",
    "landmarks/lift_39_report.json",
    "control/vgt_control_map.json",
    "control/target_control_keypoint_motion.json",
    "control/projected_control_keypoint_motion.json",
    "motion/vgt_motion.npz",
    "motion/vgt_motion_manifest.json",
    "previews/vgt_three_view.png",
    "previews/vgt_isometric.png",
    "previews/vgt_motion_30fps.mp4",
    "reports/reachability_report.json",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_artifact(case_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest artifact path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = case_root / path
    return path.resolve()


def _video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if info["width"] <= 0 or info["height"] <= 0:
        raise ValueError(f"video has invalid dimensions: {path}")
    return info


def _all_finite_json(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _check_record(
    checks: dict,
    name: str,
    function: Callable[[], dict | str | None],
) -> None:
    try:
        evidence = function()
        checks[name] = {"passed": True}
        if isinstance(evidence, dict):
            checks[name].update(evidence)
        elif isinstance(evidence, str):
            checks[name]["evidence"] = evidence
    except Exception as error:
        checks[name] = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def verify_case(case_root: Path) -> dict:
    """Run named acceptance checks and always publish a final JSON report."""
    root = Path(case_root).resolve()
    checks: dict[str, dict] = {}

    def required_files() -> dict:
        missing = [
            relative
            for relative in REQUIRED_FILES
            if not (root / relative).is_file()
        ]
        if missing:
            raise ValueError("missing required files: " + ", ".join(missing))
        return {"file_count": len(REQUIRED_FILES)}

    _check_record(checks, "required_files", required_files)

    json_paths = [
        root / relative
        for relative in REQUIRED_FILES
        if relative.endswith(".json") and (root / relative).is_file()
    ]
    loaded: dict[str, dict] = {}

    def strict_json() -> dict:
        for path in json_paths:
            payload = _load_json(path)
            if not _all_finite_json(payload):
                raise ValueError(f"JSON contains a non-finite number: {path}")
            loaded[str(path.relative_to(root))] = payload
        return {"json_file_count": len(loaded)}

    _check_record(checks, "strict_finite_json", strict_json)

    def payload(relative: str) -> dict:
        key = str(Path(relative))
        if key not in loaded:
            loaded[key] = _load_json(root / relative)
        return loaded[key]

    def input_sources() -> dict:
        manifest = payload("input_manifest.json")
        sources = manifest.get("sources")
        if not isinstance(sources, dict) or len(sources) < 4:
            raise ValueError("input manifest must contain four hashed sources")
        verified = 0
        for label, item in sources.items():
            if not isinstance(item, dict):
                raise ValueError(f"input source {label!r} is malformed")
            path = _resolved_artifact(root, item.get("path"))
            if not path.is_file() or _sha256(path) != item.get("sha256"):
                raise ValueError(f"input source hash mismatch: {label}")
            verified += 1
        video = manifest.get("video")
        if (
            not isinstance(video, dict)
            or int(video.get("frame_count", -1)) != EXPECTED_FRAMES
            or not math.isclose(
                float(video.get("fps", 0.0)),
                EXPECTED_FPS,
                abs_tol=0.01,
            )
        ):
            raise ValueError("input video must declare 272 frames at 30 FPS")
        mesh_provenance = manifest.get("mesh_provenance")
        generation_method = manifest.get("mesh_generation_method")
        if isinstance(mesh_provenance, dict):
            generation_method = mesh_provenance.get("generation_method")
        if generation_method != "hunyuan3d_from_video_frame":
            raise ValueError("Hunyuan3D mesh provenance is missing")
        return {"verified_source_hashes": verified}

    _check_record(checks, "input_manifest_and_hashes", input_sources)

    def observation_roles() -> dict:
        trajectory = payload(
            "observation/keypoint_trajectory_2d_39.json"
        )
        if trajectory.get("schema") != "qianji.keypoint_trajectory_2d_39":
            raise ValueError("unsupported 39-point observation schema")
        if tuple(trajectory.get("roles", ())) != SUPERANIMAL_QUADRUPED_39:
            raise ValueError("observation role list is not the exact 39 roles")
        frames = trajectory.get("frames")
        if not isinstance(frames, list) or len(frames) != EXPECTED_FRAMES:
            raise ValueError("observation must contain exactly 272 frames")
        for frame_idx, frame in enumerate(frames):
            if frame.get("frame_idx") != frame_idx:
                raise ValueError("observation frames are not contiguous")
            if tuple(frame.get("keypoints", {})) != SUPERANIMAL_QUADRUPED_39:
                raise ValueError(f"frame {frame_idx} does not contain 39 roles")
        quality = payload("observation/keypoint_39_quality_report.json")
        if (
            quality.get("frame_count") != EXPECTED_FRAMES
            or quality.get("role_count") != 39
        ):
            raise ValueError("quality report counts differ")
        if quality.get("interpolation_applied", False) is not False:
            raise ValueError("observation interpolation is forbidden")
        return {"frames": EXPECTED_FRAMES, "roles": 39}

    _check_record(checks, "observation_39_roles", observation_roles)

    def neutral_and_motion() -> dict:
        neutral = payload("landmarks/neutral_landmarks_39.json")
        motion = payload("landmarks/keypoint_motion_3d_39.json")
        report = payload("landmarks/lift_39_report.json")
        if tuple(neutral.get("landmarks", {})) != SUPERANIMAL_QUADRUPED_39:
            raise ValueError("neutral landmarks do not contain exact 39 roles")
        if motion.get("schema") != "qianji-keypoint-trajectory-39-v1":
            raise ValueError("unsupported 39-point 3D motion schema")
        if motion.get("reconstruction_kind", report.get("reconstruction_kind")) != (
            "body_relative_2_5d_retarget"
        ):
            raise ValueError("3D motion must be labeled body-relative 2.5D")
        frames = motion.get("frames")
        if not isinstance(frames, list) or len(frames) != EXPECTED_FRAMES:
            raise ValueError("3D motion must contain exactly 272 frames")
        for frame_idx, frame in enumerate(frames):
            if tuple(frame.get("keypoints", {})) != SUPERANIMAL_QUADRUPED_39:
                raise ValueError(f"3D frame {frame_idx} does not contain 39 roles")
            for role, value in frame["keypoints"].items():
                vector = np.asarray(value, dtype=float)
                if vector.shape != (4,) or not np.isfinite(vector).all():
                    raise ValueError(
                        f"3D frame {frame_idx} role {role} is not finite XYZC"
                    )
        if report.get("interpolation_applied", False) is not False:
            raise ValueError("3D interpolation is forbidden")
        return {"frames": EXPECTED_FRAMES, "roles": 39}

    _check_record(checks, "body_relative_3d_39", neutral_and_motion)

    def control_boundary() -> dict:
        control = payload("control/vgt_control_map.json")
        controls = control.get("controls")
        if (
            control.get("schema") != "qianji.vgt_control_map"
            or not isinstance(controls, dict)
            or tuple(controls) != KEYPOINT_ROLES
        ):
            raise ValueError("VGT control map must contain exact six controls")
        sites = []
        for role, item in controls.items():
            if not isinstance(item.get("observation_primary"), str):
                raise ValueError(f"control {role} has no observation primary")
            if not isinstance(item.get("site"), str):
                raise ValueError(f"control {role} has no structural site")
            sites.append(item["site"])
        if len(set(sites)) != 6:
            raise ValueError("six control roles must map to distinct sites")
        return {"observation_roles": 39, "control_roles": 6}

    _check_record(checks, "observation_control_split", control_boundary)

    robot: dict | None = None

    def robot_contract() -> dict:
        nonlocal robot
        robot = payload("initial_model/robot.json")
        sites = robot.get("sites")
        rods = robot.get("rod_groups")
        if not isinstance(sites, dict) or len(sites) != EXPECTED_SITES:
            raise ValueError("robot must contain exactly 12 sites")
        if not isinstance(rods, list) or len(rods) != EXPECTED_RODS:
            raise ValueError("robot must contain exactly 30 rods")
        for rod in rods:
            if (
                rod.get("site1") not in sites
                or rod.get("site2") not in sites
                or rod.get("site1") == rod.get("site2")
            ):
                raise ValueError("rod endpoint does not reference two robot sites")
        return {"sites": 12, "rods": 30}

    _check_record(checks, "robot_12_sites_30_rods", robot_contract)

    sequence = None

    def vgt_sequence_contract() -> dict:
        nonlocal sequence, robot
        if robot is None:
            robot = payload("initial_model/robot.json")
        sequence = load_and_validate_vgt_sequence(
            root / "motion/vgt_motion.npz",
            robot,
            expected_frames=EXPECTED_FRAMES,
            expected_sites=EXPECTED_SITES,
            expected_rods=EXPECTED_RODS,
        )
        if not np.allclose(
            np.diff(sequence.times),
            1.0 / EXPECTED_FPS,
            atol=1e-6,
            rtol=1e-5,
        ):
            raise ValueError("VGT sequence times are not 30 FPS")
        return {"positions_shape": [272, 12, 3], "finite": True}

    _check_record(checks, "vgt_npz_contract", vgt_sequence_contract)

    def manifest_hashes() -> dict:
        manifest = payload("motion/vgt_motion_manifest.json")
        records = (
            "vgt_motion",
            "robot",
            "rig",
            "desired_control_motion",
            "projected_control_motion",
        )
        verified = 0
        for name in records:
            item = manifest.get(name)
            if not isinstance(item, dict):
                raise ValueError(f"VGT manifest is missing {name}")
            path = _resolved_artifact(root, item.get("path"))
            if not path.is_file() or _sha256(path) != item.get("sha256"):
                raise ValueError(f"VGT manifest hash mismatch: {name}")
            verified += 1
        if (
            manifest.get("positions_shape") != [272, 12, 3]
            or manifest.get("site_count") != 12
            or manifest.get("rod_count") != 30
            or not math.isclose(float(manifest.get("fps", 0.0)), 30.0)
        ):
            raise ValueError("VGT manifest counts/rate differ")
        return {"verified_hashes": verified}

    _check_record(checks, "manifest_hashes", manifest_hashes)

    def distinct_targets() -> dict:
        manifest = payload("motion/vgt_motion_manifest.json")
        desired = manifest.get("desired_control_motion", {})
        projected = manifest.get("projected_control_motion", {})
        desired_path = _resolved_artifact(root, desired.get("path"))
        projected_path = _resolved_artifact(root, projected.get("path"))
        if desired_path == projected_path:
            raise ValueError("desired and projected paths are aliases")
        if desired.get("sha256") == projected.get("sha256"):
            raise ValueError("desired and projected contents are identical")
        return {
            "desired_sha256": desired.get("sha256"),
            "projected_sha256": projected.get("sha256"),
        }

    _check_record(checks, "desired_projected_distinct", distinct_targets)

    def video_contract() -> dict:
        observation_info = _video_info(
            root / "observation/keypoint_39_preview.mp4"
        )
        vgt_info = _video_info(root / "previews/vgt_motion_30fps.mp4")
        for label, info in (
            ("observation", observation_info),
            ("VGT", vgt_info),
        ):
            if info["frame_count"] != EXPECTED_FRAMES or not math.isclose(
                info["fps"],
                EXPECTED_FPS,
                abs_tol=0.01,
            ):
                raise ValueError(f"{label} preview is not 272 frames at 30 FPS")
        for relative in (
            "previews/vgt_three_view.png",
            "previews/vgt_isometric.png",
        ):
            image = cv2.imread(str(root / relative))
            if image is None or image.size == 0 or not np.any(image < 245):
                raise ValueError(f"static preview is blank: {relative}")
        return {
            "observation": observation_info,
            "vgt": vgt_info,
        }

    _check_record(checks, "preview_rate_frames_and_pixels", video_contract)

    def scientific_limits() -> dict:
        input_limits = payload("input_manifest.json").get("scientific_limits")
        motion_limits = payload(
            "motion/vgt_motion_manifest.json"
        ).get("scientific_limits")
        required_input = ("metric_depth_observed",)
        required_motion = (
            "metric_depth_observed",
            "camera_calibrated",
            "global_translation_preserved",
            "dynamics_simulated",
            "mesh_vertices_deformed",
        )
        if not isinstance(input_limits, dict) or any(
            input_limits.get(name) is not False for name in required_input
        ):
            raise ValueError("input scientific limitations are incomplete")
        mesh_deformed = input_limits.get(
            "mesh_vertices_deformed",
            input_limits.get("triangle_mesh_deformed"),
        )
        if mesh_deformed is not False:
            raise ValueError("input mesh-deformation limitation is missing")
        if input_limits.get("mesh_reconstructed_in_pipeline", False) is not False:
            raise ValueError("input mesh reconstruction limitation is invalid")
        if input_limits.get("dynamics_simulated", False) is not False:
            raise ValueError("input dynamics limitation is invalid")
        if not isinstance(motion_limits, dict) or any(
            motion_limits.get(name) is not False for name in required_motion
        ):
            raise ValueError("motion scientific limitations are incomplete")
        return {"depth_claim": False, "dynamics_claim": False}

    _check_record(checks, "scientific_limits", scientific_limits)

    def reachability() -> dict:
        report = payload("reports/reachability_report.json")
        selected = report.get("selected_candidate")
        if not isinstance(selected, dict) or selected.get("eligible") is not True:
            raise ValueError("reachability report has no eligible selection")
        summary = selected.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("selected reachability summary is missing")
        status = summary.get("status_counts")
        if (
            summary.get("frames") != EXPECTED_FRAMES
            or not isinstance(status, dict)
            or status.get("unreachable") != 0
            or float(summary.get("max_keypoint_error_m", math.inf)) > 0.05
        ):
            raise ValueError("selected reachability misses acceptance thresholds")
        return {
            "candidate_id": selected.get("candidate_id"),
            "status_counts": status,
            "max_keypoint_error_m": summary.get("max_keypoint_error_m"),
        }

    _check_record(checks, "reachability_accepted", reachability)

    passed = all(item["passed"] for item in checks.values())
    output_hashes = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        if path.is_file():
            output_hashes[relative] = _sha256(path)
    report = {
        "schema": "qianji.39point_vgt_final_acceptance",
        "schema_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_root": str(root),
        "passed": passed,
        "checks": checks,
        "output_hashes": output_hashes,
        "exact_commands": [
            "uv run --python 3.12 --with '.[dev]' python -m pytest -q",
            f"python experiments/39point_vgt_cat/verify_case.py {root}",
        ],
    }
    report_path = root / "reports/final_acceptance_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_case(args.case_root)
    print(args.case_root / "reports/final_acceptance_report.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
