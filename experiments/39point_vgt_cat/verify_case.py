#!/usr/bin/env python3
"""Verify every artifact and scientific boundary of a 39-point VGT case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from qianji_animal_motion.keypoints_39 import validate_39point_observation
from qianji_animal_motion.lift_39 import (
    validate_lifted_39_motion,
    validate_neutral_landmarks_39,
)
from qianji_animal_motion.lift_3d import (
    KEYPOINT_ROLES,
    _validate_trajectory,
)
from qianji_animal_motion.vgt_control import (
    infer_uniform_contraction_fraction,
    validate_vgt_control_map,
)
from qianji_animal_motion.vgt_sequence import (
    compute_rod_constraint_metrics,
    load_and_validate_vgt_sequence,
    validate_control_pair_against_sequence,
)


EXPECTED_FRAMES = 272
EXPECTED_FPS = 30.0
EXPECTED_SITES = 12
EXPECTED_RODS = 30
MAX_KEYPOINT_ERROR_M = 0.05
MAX_EDGE_VIOLATION_M = 0.0005
MAX_CLIPPED_FRACTION = 0.05

REQUIRED_FILES = (
    "input_manifest.json",
    "observation/keypoint_trajectory_2d_39.json",
    "observation/keypoint_39_quality_report.json",
    "observation/keypoint_39_preview.mp4",
    "initial_model/robot_base.json",
    "initial_model/robot.json",
    "initial_model/robot.xml",
    "initial_model/robot_scene.xml",
    "initial_model/robot_conversion_manifest.json",
    "initial_model/rig_bbox.json",
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
    "reports/selected_candidate.json",
    "reports/experiment_provenance.json",
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


def _resolved_artifact(
    case_root: Path,
    value: object,
    *,
    require_relative: bool = False,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a non-empty string")
    raw = Path(value)
    if require_relative and raw.is_absolute():
        raise ValueError(f"case artifact path must be relative: {value}")
    path = raw if raw.is_absolute() else case_root / raw
    return path.resolve()


def _verified_record(
    case_root: Path,
    item: object,
    label: str,
    *,
    require_relative: bool,
) -> Path:
    if not isinstance(item, dict):
        raise ValueError(f"artifact record {label!r} is missing")
    path = _resolved_artifact(
        case_root,
        item.get("path"),
        require_relative=require_relative,
    )
    digest = item.get("sha256")
    if (
        not path.is_file()
        or not isinstance(digest, str)
        or len(digest) != 64
        or _sha256(path) != digest
    ):
        raise ValueError(f"artifact hash mismatch: {label}")
    return path


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
    if (
        info["width"] <= 0
        or info["height"] <= 0
        or not math.isfinite(info["fps"])
        or info["fps"] <= 0.0
    ):
        raise ValueError(f"video has invalid metadata: {path}")
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
    loaded: dict[str, dict] = {}

    def payload(relative: str) -> dict:
        key = Path(relative).as_posix()
        if key not in loaded:
            loaded[key] = _load_json(root / relative)
        return loaded[key]

    def required_files() -> dict:
        missing = [
            relative
            for relative in REQUIRED_FILES
            if not (root / relative).is_file()
            or (root / relative).stat().st_size <= 0
        ]
        if missing:
            raise ValueError("missing required files: " + ", ".join(missing))
        return {"file_count": len(REQUIRED_FILES)}

    _check_record(checks, "required_files", required_files)

    def strict_json() -> dict:
        json_paths = [
            root / relative
            for relative in REQUIRED_FILES
            if relative.endswith(".json") and (root / relative).is_file()
        ]
        for path in json_paths:
            value = _load_json(path)
            if not _all_finite_json(value):
                raise ValueError(f"JSON contains a non-finite number: {path}")
            loaded[path.relative_to(root).as_posix()] = value
        return {"json_file_count": len(json_paths)}

    _check_record(checks, "strict_finite_json", strict_json)

    source_paths: dict[str, Path] = {}

    def input_sources() -> dict:
        manifest = payload("input_manifest.json")
        if (
            manifest.get("schema") != "qianji.cat_39point_input_manifest"
            or manifest.get("schema_version") != "0.1.0"
            or manifest.get("reference_frame") != 152
        ):
            raise ValueError("input manifest header is invalid")
        sources = manifest.get("sources")
        expected_labels = ("video", "predictions", "mesh", "corrected_spine")
        if not isinstance(sources, dict) or set(sources) != set(
            expected_labels
        ):
            raise ValueError("input manifest must contain exact source records")
        for label in expected_labels:
            source_paths[label] = _verified_record(
                root,
                sources[label],
                f"input {label}",
                require_relative=False,
            )
        video = manifest.get("video")
        if not isinstance(video, dict):
            raise ValueError("input manifest video metadata is missing")
        observed_video = _video_info(source_paths["video"])
        for field in ("width", "height", "frame_count"):
            if video.get(field) != observed_video[field]:
                raise ValueError(f"input video {field} differs from source")
        if (
            video.get("frame_count") != EXPECTED_FRAMES
            or not math.isclose(
                float(video.get("fps")),
                observed_video["fps"],
                rel_tol=0.0,
                abs_tol=0.01,
            )
            or not math.isclose(
                float(video.get("fps")),
                EXPECTED_FPS,
                rel_tol=0.0,
                abs_tol=0.01,
            )
        ):
            raise ValueError("input video must be 272 frames at 30 FPS")
        corrected = _load_json(source_paths["corrected_spine"])
        _validate_trajectory(corrected)
        if corrected["video"] != video:
            raise ValueError("corrected trajectory video metadata differs")
        identity = manifest.get("identity_anchor")
        if (
            not isinstance(identity, dict)
            or identity.get("frame_idx") != 152
            or identity.get("confirmed_by") != "manual"
            or identity.get("front_assignment") not in {"keep", "swap"}
            or identity.get("rear_assignment") not in {"keep", "swap"}
        ):
            raise ValueError("input identity anchor is not manually confirmed")
        corrected_anchor = corrected.get("identity_anchor")
        if not isinstance(corrected_anchor, dict) or any(
            corrected_anchor.get(field) != identity[field]
            for field in (
                "frame_idx",
                "front_assignment",
                "rear_assignment",
                "confirmed_by",
            )
        ):
            raise ValueError("manifest identity anchor differs from corrected input")
        mesh_provenance = manifest.get("mesh_provenance")
        if (
            not isinstance(mesh_provenance, dict)
            or mesh_provenance.get("generation_method")
            != "hunyuan3d_from_video_frame"
        ):
            raise ValueError("Hunyuan3D mesh provenance is missing")
        return {"verified_source_hashes": len(source_paths)}

    _check_record(checks, "input_manifest_and_hashes", input_sources)

    observation: dict | None = None

    def observation_roles() -> dict:
        nonlocal observation
        observation = payload("observation/keypoint_trajectory_2d_39.json")
        quality = payload("observation/keypoint_39_quality_report.json")
        evidence = validate_39point_observation(
            observation,
            quality,
            expected_frames=EXPECTED_FRAMES,
            require_source_lineage=True,
        )
        lineage = observation["source_lineage"]
        manifest = payload("input_manifest.json")
        expected = {
            "video_sha256": manifest["sources"]["video"]["sha256"],
            "predictions_sha256": manifest["sources"]["predictions"]["sha256"],
            "corrected_trajectory_sha256": manifest["sources"][
                "corrected_spine"
            ]["sha256"],
        }
        if lineage != expected:
            raise ValueError("39-point source lineage differs from input manifest")
        identity = observation.get("identity_anchor")
        if identity != {
            key: manifest["identity_anchor"][key]
            for key in ("frame_idx", "front_assignment", "rear_assignment")
        }:
            raise ValueError("39-point identity anchor differs from manifest")
        return evidence

    _check_record(checks, "observation_39_roles", observation_roles)

    neutral: dict | None = None
    motion_39: dict | None = None

    def neutral_and_motion() -> dict:
        nonlocal neutral, motion_39, observation
        if observation is None:
            observation = payload(
                "observation/keypoint_trajectory_2d_39.json"
            )
        neutral = payload("landmarks/neutral_landmarks_39.json")
        motion_39 = payload("landmarks/keypoint_motion_3d_39.json")
        lift_report = payload("landmarks/lift_39_report.json")
        validate_neutral_landmarks_39(neutral)
        evidence = validate_lifted_39_motion(
            motion_39,
            lift_report,
            observation=observation,
            neutral_landmarks=neutral,
            expected_frames=EXPECTED_FRAMES,
        )
        neutral_sources = neutral.get("sources")
        report_sources = lift_report.get("sources")
        expected_labels = (
            "trajectory_39",
            "corrected_spine",
            "robot",
            "rig",
        )
        if (
            not isinstance(neutral_sources, dict)
            or set(neutral_sources) != set(expected_labels)
            or report_sources != neutral_sources
        ):
            raise ValueError("lift source records are missing or inconsistent")
        verified = {
            label: _verified_record(
                root,
                neutral_sources[label],
                f"lift source {label}",
                require_relative=label != "corrected_spine",
            )
            for label in expected_labels
        }
        expected_paths = {
            "trajectory_39": (
                root / "observation/keypoint_trajectory_2d_39.json"
            ).resolve(),
            "corrected_spine": source_paths.get("corrected_spine"),
            "robot": (root / "initial_model/robot_base.json").resolve(),
            "rig": (root / "initial_model/rig_bbox.json").resolve(),
        }
        if expected_paths["corrected_spine"] is None:
            expected_paths["corrected_spine"] = _verified_record(
                root,
                payload("input_manifest.json").get("sources", {}).get(
                    "corrected_spine"
                ),
                "input corrected_spine",
                require_relative=False,
            )
        if verified != expected_paths:
            raise ValueError("lift sources do not match canonical parent artifacts")
        evidence["verified_source_hashes"] = len(verified)
        return evidence

    _check_record(checks, "body_relative_3d_39", neutral_and_motion)

    robot: dict | None = None
    rig: dict | None = None
    selected: dict | None = None

    def robot_contract() -> dict:
        nonlocal robot, rig, selected
        robot = payload("initial_model/robot.json")
        base_robot = payload("initial_model/robot_base.json")
        rig = payload("initial_model/rig_keypoints.json")
        selected = payload("reports/selected_candidate.json")
        sites = robot.get("sites")
        rods = robot.get("rod_groups")
        if not isinstance(sites, dict) or len(sites) != EXPECTED_SITES:
            raise ValueError("canonical robot must contain exactly 12 sites")
        if not isinstance(rods, list) or len(rods) != EXPECTED_RODS:
            raise ValueError("canonical robot must contain exactly 30 rods")
        if (
            not isinstance(base_robot.get("sites"), dict)
            or len(base_robot["sites"]) != EXPECTED_SITES
            or not isinstance(base_robot.get("rod_groups"), list)
            or len(base_robot["rod_groups"]) != EXPECTED_RODS
        ):
            raise ValueError("base robot contract is invalid")
        for rod in rods:
            if (
                not isinstance(rod, dict)
                or rod.get("site1") not in sites
                or rod.get("site2") not in sites
                or rod.get("site1") == rod.get("site2")
            ):
                raise ValueError("canonical rod endpoints are invalid")
        if (
            rig.get("schema") != "qianji-key-site-map"
            or not isinstance(rig.get("key_site_map"), dict)
            or tuple(rig["key_site_map"]) != KEYPOINT_ROLES
            or len(set(rig["key_site_map"].values())) != len(KEYPOINT_ROLES)
            or any(site not in sites for site in rig["key_site_map"].values())
        ):
            raise ValueError("canonical rig contract is invalid")
        artifacts = selected.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("selected candidate artifacts are missing")
        if artifacts.get("robot", {}).get("sha256") != _sha256(
            root / "initial_model/robot.json"
        ):
            raise ValueError("canonical robot is not the selected robot")
        if artifacts.get("rig", {}).get("sha256") != _sha256(
            root / "initial_model/rig_keypoints.json"
        ):
            raise ValueError("canonical rig is not the selected rig")
        conversion = payload(
            "initial_model/robot_conversion_manifest.json"
        )
        if (
            conversion.get("schema") != "qianji.selected_robot_conversion"
            or conversion.get("schema_version") != "1.0.0"
            or conversion.get("converter")
            != "QianJi mujoco_builder/json2xml_v7_perrod.py"
        ):
            raise ValueError("selected robot conversion manifest is invalid")
        for label in ("source_robot", "robot_xml", "robot_scene_xml"):
            _verified_record(
                root,
                conversion.get("artifacts", {}).get(label),
                f"robot conversion {label}",
                require_relative=True,
            )
        xml_roots = {}
        for label, relative in (
            ("robot_xml", "initial_model/robot.xml"),
            ("robot_scene_xml", "initial_model/robot_scene.xml"),
        ):
            try:
                xml_roots[label] = ET.parse(root / relative).getroot()
            except ET.ParseError as error:
                raise ValueError(f"{label} is not valid XML") from error
            if xml_roots[label].tag != "mujoco":
                raise ValueError(f"{label} root must be mujoco")
        if (
            xml_roots["robot_xml"].get("model")
            != xml_roots["robot_scene_xml"].get("model")
        ):
            raise ValueError("robot and scene XML model names differ")
        anchors = {}
        for body in xml_roots["robot_xml"].iter("body"):
            name = body.get("name", "")
            if not name.startswith("anchor_"):
                continue
            site_name = name.removeprefix("anchor_")
            try:
                position = np.asarray(
                    [float(value) for value in body.get("pos", "").split()],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"XML anchor {site_name!r} position is malformed"
                ) from error
            if (
                site_name in anchors
                or position.shape != (3,)
                or not np.isfinite(position).all()
            ):
                raise ValueError("XML anchor positions are invalid")
            anchors[site_name] = position
        if set(anchors) != set(sites) or any(
            not np.allclose(
                anchors[name],
                np.asarray(item["pos"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
            for name, item in sites.items()
        ):
            raise ValueError("canonical XML anchors differ from selected robot")
        xml_joints = {
            joint.get("name"): joint
            for joint in xml_roots["robot_xml"].iter("joint")
            if joint.get("name")
        }
        xml_actuators = {
            actuator.get("name"): actuator
            for actuator in xml_roots["robot_xml"].iter("position")
            if actuator.get("name")
        }
        for rod in rods:
            constraint = rod.get("constraint")
            if not isinstance(constraint, dict):
                raise ValueError("canonical rod constraint is missing")
            try:
                slide_range = float(
                    constraint["slide_range_each_side_required"]
                )
                slide_min = float(
                    constraint.get("slide_min_each_side_required", 0.0)
                )
                slide_max = float(
                    constraint.get(
                        "slide_max_each_side_required",
                        slide_range,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("canonical slide limits are malformed") from error
            if (
                not np.isfinite([slide_min, slide_max, slide_range]).all()
                or slide_max < slide_min
            ):
                raise ValueError("canonical slide limits are invalid")
            for side in ("left", "right"):
                joint_name = f"{rod['name']}__slide_{side}"
                joint = xml_joints.get(joint_name)
                try:
                    joint_range = np.asarray(
                        [
                            float(value)
                            for value in joint.get("range", "").split()
                        ],
                        dtype=float,
                    )
                except (AttributeError, ValueError) as error:
                    raise ValueError(
                        f"XML joint {joint_name!r} range is malformed"
                    ) from error
                if (
                    joint_range.shape != (2,)
                    or not np.allclose(
                        joint_range,
                        [slide_min, slide_max],
                        rtol=0.0,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"XML joint {joint_name!r} limits differ from robot"
                    )
                if rod.get("actuated") is True:
                    actuator_name = f"act_{joint_name}"
                    actuator = xml_actuators.get(actuator_name)
                    try:
                        control_range = np.asarray(
                            [
                                float(value)
                                for value in actuator.get(
                                    "ctrlrange",
                                    "",
                                ).split()
                            ],
                            dtype=float,
                        )
                    except (AttributeError, ValueError) as error:
                        raise ValueError(
                            f"XML actuator {actuator_name!r} range is malformed"
                        ) from error
                    if (
                        control_range.shape != (2,)
                        or not np.allclose(
                            control_range,
                            [slide_min, slide_max],
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ):
                        raise ValueError(
                            f"XML actuator {actuator_name!r} limits "
                            "differ from robot"
                        )
        contraction = infer_uniform_contraction_fraction(robot)
        return {
            "sites": EXPECTED_SITES,
            "rods": EXPECTED_RODS,
            "actual_contraction_fraction": contraction,
        }

    _check_record(checks, "robot_12_sites_30_rods", robot_contract)

    def control_boundary() -> dict:
        nonlocal robot, rig
        if robot is None:
            robot = payload("initial_model/robot.json")
        if rig is None:
            rig = payload("initial_model/rig_keypoints.json")
        control = payload("control/vgt_control_map.json")
        return validate_vgt_control_map(control, robot, rig)

    _check_record(checks, "observation_control_split", control_boundary)

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
        expected_times = np.arange(EXPECTED_FRAMES) / EXPECTED_FPS
        if not np.allclose(
            sequence.times,
            expected_times,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("VGT sequence times are not exact 30 FPS")
        return {"positions_shape": [272, 12, 3], "finite": True}

    _check_record(checks, "vgt_npz_contract", vgt_sequence_contract)

    def manifest_hashes() -> dict:
        nonlocal sequence, robot, rig, selected
        if sequence is None:
            if robot is None:
                robot = payload("initial_model/robot.json")
            sequence = load_and_validate_vgt_sequence(
                root / "motion/vgt_motion.npz",
                robot,
                expected_frames=EXPECTED_FRAMES,
                expected_sites=EXPECTED_SITES,
                expected_rods=EXPECTED_RODS,
            )
        if rig is None:
            rig = payload("initial_model/rig_keypoints.json")
        if selected is None:
            selected = payload("reports/selected_candidate.json")
        manifest = payload("motion/vgt_motion_manifest.json")
        if (
            manifest.get("schema") != "qianji.vgt_motion_manifest"
            or manifest.get("schema_version") != "0.1.0"
            or manifest.get("frame_count") != EXPECTED_FRAMES
            or manifest.get("fps") != EXPECTED_FPS
            or manifest.get("site_count") != EXPECTED_SITES
            or manifest.get("rod_count") != EXPECTED_RODS
            or manifest.get("times_shape") != [EXPECTED_FRAMES]
            or manifest.get("positions_shape") != [EXPECTED_FRAMES, 12, 3]
            or manifest.get("site_names") != list(sequence.site_names)
        ):
            raise ValueError("VGT manifest shape/count contract is invalid")
        records = (
            "vgt_motion",
            "robot",
            "rig",
            "desired_control_motion",
            "projected_control_motion",
        )
        paths = {
            name: _verified_record(
                root,
                manifest.get(name),
                f"VGT manifest {name}",
                require_relative=True,
            )
            for name in records
        }
        if paths["robot"] != (root / "initial_model/robot.json").resolve():
            raise ValueError("VGT manifest does not use canonical robot")
        if paths["rig"] != (root / "initial_model/rig_keypoints.json").resolve():
            raise ValueError("VGT manifest does not use canonical rig")
        if manifest.get("selected_candidate") != selected:
            raise ValueError("VGT manifest selected candidate differs")
        desired = _load_json(paths["desired_control_motion"])
        projected = _load_json(paths["projected_control_motion"])
        validate_control_pair_against_sequence(
            desired,
            projected,
            sequence,
            rig,
            expected_frames=EXPECTED_FRAMES,
            expected_fps=EXPECTED_FPS,
        )
        if (
            manifest["vgt_motion"].get("source_sha256")
            != selected["artifacts"]["site_npz"]["sha256"]
            or manifest["robot"]["sha256"]
            != selected["artifacts"]["robot"]["sha256"]
            or manifest["rig"]["sha256"]
            != selected["artifacts"]["rig"]["sha256"]
            or manifest["desired_control_motion"]["sha256"]
            != selected["artifacts"]["desired_motion"]["sha256"]
            or manifest["projected_control_motion"]["sha256"]
            != selected["artifacts"]["projected_motion"]["sha256"]
        ):
            raise ValueError("VGT manifest hashes differ from selected candidate")
        return {"verified_hashes": len(records)}

    _check_record(checks, "manifest_hashes", manifest_hashes)

    def distinct_targets() -> dict:
        manifest = payload("motion/vgt_motion_manifest.json")
        desired = manifest.get("desired_control_motion")
        projected = manifest.get("projected_control_motion")
        desired_path = _resolved_artifact(
            root,
            desired.get("path") if isinstance(desired, dict) else None,
            require_relative=True,
        )
        projected_path = _resolved_artifact(
            root,
            projected.get("path") if isinstance(projected, dict) else None,
            require_relative=True,
        )
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
                rel_tol=0.0,
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
        return {"observation": observation_info, "vgt": vgt_info}

    _check_record(checks, "preview_rate_frames_and_pixels", video_contract)

    def scientific_limits() -> dict:
        input_limits = payload("input_manifest.json").get("scientific_limits")
        motion_limits = payload(
            "motion/vgt_motion_manifest.json"
        ).get("scientific_limits")
        required_input = (
            "metric_depth_observed",
            "camera_calibrated",
            "global_translation_preserved",
            "triangle_mesh_deformed",
            "mesh_vertices_deformed",
            "mesh_reconstructed_in_pipeline",
            "dynamics_simulated",
        )
        required_motion = (
            "metric_depth_observed",
            "camera_calibrated",
            "global_translation_preserved",
            "dynamics_simulated",
            "mesh_vertices_deformed",
        )
        if not isinstance(input_limits, dict) or any(
            name not in input_limits or input_limits[name] is not False
            for name in required_input
        ):
            raise ValueError("input scientific limitations are incomplete")
        if not isinstance(motion_limits, dict) or any(
            name not in motion_limits or motion_limits[name] is not False
            for name in required_motion
        ):
            raise ValueError("motion scientific limitations are incomplete")
        return {"depth_claim": False, "dynamics_claim": False}

    _check_record(checks, "scientific_limits", scientific_limits)

    def reachability() -> dict:
        nonlocal sequence, robot, selected
        report = payload("reports/reachability_report.json")
        selected_report = report.get("selected_candidate")
        if selected is None:
            selected = payload("reports/selected_candidate.json")
        if selected_report != selected or selected.get("eligible") is not True:
            raise ValueError("reachability selection is inconsistent")
        if (
            report.get("schema")
            != "qianji.39point_reachability_comparison"
            or report.get("schema_version") != "0.1.0"
            or not isinstance(report.get("candidates"), list)
            or selected not in report["candidates"]
        ):
            raise ValueError("reachability comparison contract is invalid")
        candidate_root = selected.get("candidate_root")
        if (
            not isinstance(candidate_root, str)
            or not candidate_root
            or Path(candidate_root).is_absolute()
            or not (root / candidate_root).is_dir()
        ):
            raise ValueError("selected candidate root must be case-relative")
        if selected.get("eligibility_failures") != []:
            raise ValueError("selected candidate contains eligibility failures")
        acceptance = report.get("acceptance")
        if acceptance != {
            "unreachable_frames": 0,
            "max_keypoint_error_m": MAX_KEYPOINT_ERROR_M,
            "max_edge_violation_m": MAX_EDGE_VIOLATION_M,
            "max_estimated_clipped_fraction": MAX_CLIPPED_FRACTION,
            "max_recomputed_violated_rod_fraction": MAX_CLIPPED_FRACTION,
            "sites": EXPECTED_SITES,
            "rods": EXPECTED_RODS,
        }:
            raise ValueError("reachability acceptance thresholds differ")
        summary = selected.get("summary")
        status = summary.get("status_counts") if isinstance(summary, dict) else None
        if (
            not isinstance(summary, dict)
            or summary.get("frames") != EXPECTED_FRAMES
            or not isinstance(status, dict)
            or set(status) != {"feasible", "marginal", "unreachable"}
            or sum(status.values()) != EXPECTED_FRAMES
            or status["unreachable"] != 0
            or float(summary["max_keypoint_error_m"]) > MAX_KEYPOINT_ERROR_M
            or float(summary["max_edge_violation_m"]) > MAX_EDGE_VIOLATION_M
            or float(summary["max_estimated_clipped_fraction"])
            > MAX_CLIPPED_FRACTION
        ):
            raise ValueError("selected reachability misses acceptance thresholds")
        artifacts = selected.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("selected reachability artifacts are missing")
        for name, item in artifacts.items():
            _verified_record(
                root,
                item,
                f"selected {name}",
                require_relative=True,
            )
        qianji_report = _load_json(
            _resolved_artifact(
                root,
                artifacts["reachability_report"]["path"],
                require_relative=True,
            )
        )
        qianji_frames = qianji_report.get("frames")
        if (
            qianji_report.get("summary") != summary
            or qianji_report.get("extension_only")
            is not selected.get("extension_only")
            or not isinstance(qianji_frames, list)
            or len(qianji_frames) != EXPECTED_FRAMES
        ):
            raise ValueError("selected QianJi reachability report differs")
        desired = _load_json(
            _resolved_artifact(
                root,
                artifacts["desired_motion"]["path"],
                require_relative=True,
            )
        )
        projected = _load_json(
            _resolved_artifact(
                root,
                artifacts["projected_motion"]["path"],
                require_relative=True,
            )
        )
        observed_status = Counter()
        frame_keypoint_errors = []
        frame_edge_violations = []
        frame_clipped_fractions = []
        for frame_idx, frame in enumerate(qianji_frames):
            if (
                not isinstance(frame, dict)
                or frame.get("frame") != frame_idx
                or frame.get("status")
                not in {"feasible", "marginal", "unreachable"}
            ):
                raise ValueError("QianJi reachability frame identity is invalid")
            try:
                frame_time = float(frame["time"])
                keypoint_error = float(frame["max_keypoint_error_m"])
                edge_violation = float(frame["max_edge_violation_m"])
                clipped_fraction = float(
                    frame["estimated_clipped_fraction"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "QianJi reachability frame metrics are malformed"
                ) from error
            if (
                not np.isfinite(
                    [
                        frame_time,
                        keypoint_error,
                        edge_violation,
                        clipped_fraction,
                    ]
                ).all()
                or not math.isclose(
                    frame_time,
                    frame_idx / EXPECTED_FPS,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or min(
                    keypoint_error,
                    edge_violation,
                    clipped_fraction,
                )
                < 0.0
            ):
                raise ValueError("QianJi reachability frame metrics are invalid")
            target_points = frame.get("target_keypoints")
            projected_points = frame.get("projected_keypoints")
            if (
                not isinstance(target_points, dict)
                or set(target_points) != set(KEYPOINT_ROLES)
                or not isinstance(projected_points, dict)
                or set(projected_points) != set(KEYPOINT_ROLES)
            ):
                raise ValueError(
                    "QianJi reachability frame control roles are invalid"
                )
            for role in KEYPOINT_ROLES:
                target = np.asarray(target_points[role], dtype=float)
                expected_target = np.asarray(
                    desired["frames"][frame_idx]["keypoints"][role],
                    dtype=float,
                )
                actual_projected = np.asarray(
                    projected_points[role],
                    dtype=float,
                )
                expected_projected = np.asarray(
                    projected["frames"][frame_idx]["keypoints"][role],
                    dtype=float,
                )
                if (
                    target.shape != (4,)
                    or expected_target.shape != (4,)
                    or actual_projected.shape != (4,)
                    or expected_projected.shape != (4,)
                    or not np.isfinite(
                        np.concatenate(
                            (
                                target,
                                expected_target,
                                actual_projected,
                                expected_projected,
                            )
                        )
                    ).all()
                    or not np.allclose(
                        target[:3],
                        expected_target[:3],
                        rtol=0.0,
                        atol=1e-9,
                    )
                    or not np.allclose(
                        actual_projected,
                        expected_projected,
                        rtol=0.0,
                        atol=1e-9,
                    )
                ):
                    raise ValueError(
                        "QianJi reachability controls differ from artifacts"
                    )
            observed_status[frame["status"]] += 1
            frame_keypoint_errors.append(keypoint_error)
            frame_edge_violations.append(edge_violation)
            frame_clipped_fractions.append(clipped_fraction)
        if (
            dict(observed_status) != {
                name: status[name]
                for name in ("feasible", "marginal", "unreachable")
                if status[name]
            }
            or not math.isclose(
                max(frame_keypoint_errors),
                float(summary["max_keypoint_error_m"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                max(frame_edge_violations),
                float(summary["max_edge_violation_m"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                max(frame_clipped_fractions),
                float(summary["max_estimated_clipped_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("QianJi frame metrics do not reproduce summary")
        if robot is None:
            robot = payload("initial_model/robot.json")
        if sequence is None:
            sequence = load_and_validate_vgt_sequence(
                root / "motion/vgt_motion.npz",
                robot,
                expected_frames=EXPECTED_FRAMES,
                expected_sites=EXPECTED_SITES,
                expected_rods=EXPECTED_RODS,
            )
        geometry = compute_rod_constraint_metrics(sequence, robot)
        if (
            geometry["max_edge_violation_m"] > MAX_EDGE_VIOLATION_M
            or geometry["max_violated_rod_fraction"] > MAX_CLIPPED_FRACTION
            or selected.get("recomputed_geometry") != geometry
        ):
            raise ValueError("recomputed rod constraints miss acceptance thresholds")
        actual_contraction = infer_uniform_contraction_fraction(robot)
        if (
            not math.isclose(
                float(selected.get("contraction_fraction")),
                actual_contraction,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(selected.get("actual_contraction_fraction")),
                actual_contraction,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("selected contraction metadata differs from robot")
        return {
            "candidate_id": selected.get("candidate_id"),
            "status_counts": status,
            "max_keypoint_error_m": summary["max_keypoint_error_m"],
            "max_edge_violation_m": geometry["max_edge_violation_m"],
            "max_violated_rod_fraction": geometry[
                "max_violated_rod_fraction"
            ],
        }

    _check_record(checks, "reachability_accepted", reachability)

    def provenance() -> dict:
        record = payload("reports/experiment_provenance.json")
        if (
            record.get("schema") != "qianji.experiment_provenance"
            or record.get("schema_version") != "1.0.0"
        ):
            raise ValueError("experiment provenance header is invalid")
        for repository in ("repository", "qianji"):
            item = record.get(repository)
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("commit"), str)
                or len(item["commit"]) != 40
                or not isinstance(item.get("dirty"), bool)
                or not isinstance(item.get("status_porcelain"), list)
                or not isinstance(item.get("worktree_state_sha256"), str)
                or len(item["worktree_state_sha256"]) != 64
            ):
                raise ValueError(f"{repository} provenance is incomplete")
        scripts = record.get("scripts")
        invocation = record.get("invocation")
        environment = record.get("environment")
        if (
            not isinstance(scripts, list)
            or len(scripts) < 9
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
                for item in scripts
            )
            or not isinstance(invocation, dict)
            or not isinstance(invocation.get("argv"), list)
            or not invocation["argv"]
            or any(
                not isinstance(argument, str) or not argument
                for argument in invocation["argv"]
            )
            or not isinstance(invocation.get("working_directory"), str)
            or not invocation["working_directory"]
            or not isinstance(environment, dict)
            or not isinstance(environment.get("python_version"), str)
        ):
            raise ValueError("script/environment provenance is incomplete")
        packages = environment.get("packages")
        if not isinstance(packages, dict) or any(
            package not in packages
            or (
                packages[package] is not None
                and not isinstance(packages[package], str)
            )
            for package in ("mujoco", "numpy", "pandas", "scipy")
        ):
            raise ValueError("Python package provenance is incomplete")
        return {
            "repository_commit": record["repository"]["commit"],
            "qianji_commit": record["qianji"]["commit"],
            "qianji_dirty": record["qianji"]["dirty"],
            "invocation": invocation["argv"],
        }

    _check_record(checks, "experiment_provenance", provenance)

    passed = all(item["passed"] for item in checks.values())
    output_hashes = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        if path.is_file():
            output_hashes[relative] = _sha256(path)
    report = {
        "schema": "qianji.39point_vgt_final_acceptance",
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_id": root.name,
        "passed": passed,
        "checks": checks,
        "output_hashes": output_hashes,
        "exact_commands": [
            "PYTHONPATH=src .venv/bin/pytest",
            (
                "PYTHONPATH=src .venv/bin/python "
                "experiments/39point_vgt_cat/verify_case.py <case-root>"
            ),
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
