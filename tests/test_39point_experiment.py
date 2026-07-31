from __future__ import annotations

import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from qianji_animal_motion.keypoints_39 import (
    SUPERANIMAL_QUADRUPED_39,
    build_39point_observation,
)
from qianji_animal_motion.lift_39 import (
    build_neutral_landmarks_39,
    lift_39point_trajectory,
)
from qianji_animal_motion.lift_3d import KEYPOINT_ROLES
from qianji_animal_motion.semantic_mapping import VideoInfo
from qianji_animal_motion.vgt_control import (
    build_target_control_motion,
    build_vgt_control_map,
)
from qianji_animal_motion.vgt_sequence import (
    REACHABILITY_THRESHOLDS,
    load_and_validate_vgt_sequence,
    recompute_reachability_metrics,
    validate_control_motion,
)


ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "experiments" / "39point_vgt_cat" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_module = _load_script("compare_candidates")
verify_module = _load_script("verify_case")
select_candidate = compare_module.select_candidate
verify_case = verify_module.verify_case
candidate_record = compare_module._candidate_record
compare_candidates = compare_module.compare_candidates


def _candidate(
    candidate_id: str,
    *,
    scale: float,
    feasible_fraction: float,
    max_error: float,
    unreachable: int = 0,
    contraction: float = 0.0,
    rig: str = "bbox",
    morphology: str = "base",
) -> dict:
    eligible = unreachable == 0 and max_error <= 0.05
    return {
        "candidate_id": candidate_id,
        "motion_scale": scale,
        "contraction_fraction": contraction,
        "rig_variant": rig,
        "morphology_variant": morphology,
        "summary": {
            "status_counts": {
                "feasible": round(272 * feasible_fraction),
                "marginal": 272 - round(272 * feasible_fraction) - unreachable,
                "unreachable": unreachable,
            },
            "feasible_fraction": feasible_fraction,
            "max_keypoint_error_m": max_error,
        },
        "eligible": eligible,
        "eligibility_failures": [] if eligible else ["threshold"],
    }


def test_candidate_selection_applies_all_eligibility_and_priority_rules() -> None:
    report = {
        "candidates": [
            _candidate(
                "unreachable",
                scale=0.20,
                feasible_fraction=1.0,
                max_error=0.01,
                unreachable=1,
            ),
            _candidate(
                "over_error",
                scale=0.20,
                feasible_fraction=1.0,
                max_error=0.050001,
            ),
            _candidate(
                "scale_010",
                scale=0.10,
                feasible_fraction=1.0,
                max_error=0.001,
            ),
            _candidate(
                "scale_015_low_feasible",
                scale=0.15,
                feasible_fraction=0.4,
                max_error=0.001,
                contraction=0.1,
                rig="motion_informed",
                morphology="optimized",
            ),
            _candidate(
                "scale_015_best",
                scale=0.15,
                feasible_fraction=0.8,
                max_error=0.02,
                contraction=0.1,
                rig="bbox",
            ),
            _candidate(
                "scale_015_same_feasible_higher_error",
                scale=0.15,
                feasible_fraction=0.8,
                max_error=0.03,
            ),
        ]
    }

    selected = select_candidate(report)

    assert selected["candidate_id"] == "scale_015_best"
    assert selected["motion_scale"] == 0.15
    assert selected["contraction_fraction"] == 0.1
    assert selected["rig_variant"] == "bbox"
    assert selected["morphology_variant"] == "base"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_video(
    path: Path,
    *,
    frames: int = 272,
    fps: float = 30.0,
    size: tuple[int, int] = (32, 24),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    assert writer.isOpened()
    width, height = size
    for index in range(frames):
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        frame[:, index % width] = (20, 80, 180)
        writer.write(frame)
    writer.release()


def _robot() -> dict:
    positions = (
        (0.5, 0.0, 0.8),
        (-0.5, 0.0, 0.8),
        (0.5, 0.2, 0.0),
        (0.5, -0.2, 0.0),
        (-0.5, 0.2, 0.0),
        (-0.5, -0.2, 0.0),
        (-0.4, 0.0, 0.5),
        (-0.24, 0.0, 0.5),
        (-0.08, 0.0, 0.5),
        (0.08, 0.0, 0.5),
        (0.24, 0.0, 0.5),
        (0.4, 0.0, 0.5),
    )
    sites = {
        f"s{index:03d}": {"pos": list(position)}
        for index, position in enumerate(positions)
    }
    pairs = []
    names = list(sites)
    for offset in (1, 2, 3):
        for index in range(12):
            pair = tuple(sorted((names[index], names[(index + offset) % 12])))
            if pair not in pairs:
                pairs.append(pair)
    rods = [
        {"name": f"r{index:03d}", "site1": pair[0], "site2": pair[1]}
        for index, pair in enumerate(pairs[:30])
    ]
    return {"name": "cat", "sites": sites, "rod_groups": rods}


def _write_candidate_fixture(
    root: Path,
    *,
    declared_contraction: float = 0.0,
    actual_contraction: float = 0.0,
    clip_fraction: float = 0.0,
    tamper_position: bool = False,
) -> Path:
    root.mkdir(parents=True)
    robot = _robot()
    site_names = tuple(robot["sites"])
    neutral = np.asarray(
        [robot["sites"][name]["pos"] for name in site_names],
        dtype=float,
    )
    for rod in robot["rod_groups"]:
        left = neutral[site_names.index(rod["site1"])]
        right = neutral[site_names.index(rod["site2"])]
        current = float(np.linalg.norm(right - left))
        constraint = {
            "mode": "variable_length_fixed_stretch_ratio",
            "effective_current_length": current,
            "effective_min_length": (1.0 - actual_contraction) * current,
            "effective_max_length": current + 10.0,
        }
        if actual_contraction > 0.0:
            constraint.update(
                {
                    "slide_control_mode": "relative_around_initial",
                    "slide_min_each_side_required": (
                        -0.5 * actual_contraction * current
                    ),
                    "slide_max_each_side_required": 5.0,
                    "slide_range_each_side_required": 5.0,
                    "permitted_contraction_fraction": actual_contraction,
                }
            )
        rod["constraint"] = constraint
    robot["metadata"] = {
        "permitted_contraction_fraction": actual_contraction
    }
    rig = {
        "schema": "qianji-key-site-map",
        "key_site_map": {
            role: site_names[index]
            for index, role in enumerate(KEYPOINT_ROLES)
        },
    }
    times = np.arange(2, dtype=float) / 30.0
    positions = np.repeat(neutral[None, :, :], 2, axis=0)
    if tamper_position:
        positions[1, 0] = positions[1, 1]
    npz_path = root / "site_targets.npz"
    np.savez_compressed(
        npz_path,
        site_names=np.asarray(site_names),
        times=times,
        positions=positions,
    )

    projected_frames = []
    desired_frames = []
    indices = {name: index for index, name in enumerate(site_names)}
    for frame_idx, time in enumerate(times):
        projected_points = {
            role: [
                *positions[frame_idx, indices[rig["key_site_map"][role]]].tolist(),
                0.9,
            ]
            for role in KEYPOINT_ROLES
        }
        desired_points = json.loads(json.dumps(projected_points))
        desired_points["spine_front"][2] += 0.001
        projected_frames.append(
            {"time": float(time), "keypoints": projected_points}
        )
        desired_frames.append(
            {"time": float(time), "keypoints": desired_points}
        )
    desired = {
        "schema": "qianji-keypoint-trajectory-v1",
        "fps": 30.0,
        "frames": desired_frames,
    }
    projected = {
        "schema": "qianji-keypoint-trajectory-v1",
        "fps": 30.0,
        "frames": projected_frames,
    }
    paths = {
        "robot": root / "robot.json",
        "rig": root / "rig.json",
        "desired_motion": root / "desired.json",
        "projected_motion": root / "projected.json",
    }
    _write_json(paths["robot"], robot)
    _write_json(paths["rig"], rig)
    _write_json(paths["desired_motion"], desired)
    _write_json(paths["projected_motion"], projected)

    sequence = load_and_validate_vgt_sequence(
        npz_path,
        robot,
        expected_frames=2,
        expected_sites=12,
        expected_rods=30,
    )
    desired_motion = validate_control_motion(
        desired,
        expected_frames=2,
        expected_fps=30.0,
    )
    projected_motion = validate_control_motion(
        projected,
        expected_frames=2,
        expected_fps=30.0,
    )
    independent = recompute_reachability_metrics(
        desired_motion,
        projected_motion,
        sequence,
        robot,
    )
    summary = json.loads(json.dumps(independent["summary"]))
    report_frames = []
    for frame_idx, frame in enumerate(independent["frames"]):
        report_frames.append(
            {
                **frame,
                "target_keypoints": {
                    role: [*value[:3], 1.0]
                    for role, value in desired["frames"][frame_idx][
                        "keypoints"
                    ].items()
                },
                "projected_keypoints": projected["frames"][frame_idx][
                    "keypoints"
                ],
            }
        )
    if clip_fraction:
        summary["max_estimated_clipped_fraction"] = clip_fraction
        summary["mean_estimated_clipped_fraction"] = clip_fraction / 2.0
        report_frames[0]["estimated_clipped_fraction"] = clip_fraction
    run_summary = root / "run_summary.json"
    reachability = root / "reachability_report.json"
    _write_json(run_summary, {"summary": summary})
    _write_json(
        reachability,
        {
            "schema": "qianji-keypoint-reachability-report-v1",
            "extension_only": actual_contraction == 0.0,
            "thresholds": REACHABILITY_THRESHOLDS,
            "summary": summary,
            "frames": report_frames,
        },
    )
    _write_json(
        root / "candidate.json",
        {
            "candidate_id": root.name,
            "expected_frames": 2,
            "parameters": {
                "motion_scale": 0.1,
                "contraction_fraction": declared_contraction,
                "rig_variant": "bbox",
                "morphology_variant": "base",
            },
            "artifacts": {
                "run_summary": str(run_summary),
                "reachability_report": str(reachability),
                "robot": str(paths["robot"]),
                "rig": str(paths["rig"]),
                "desired_motion": str(paths["desired_motion"]),
                "projected_motion": str(paths["projected_motion"]),
                "site_npz": str(npz_path),
            },
        },
    )
    return root


def test_candidate_gates_reject_edge_violation_and_clipping(
    tmp_path: Path,
) -> None:
    edge = candidate_record(
        _write_candidate_fixture(
            tmp_path / "edge",
            tamper_position=True,
        )
    )
    with pytest.raises(ValueError, match="independently recomputed"):
        candidate_record(
            _write_candidate_fixture(
                tmp_path / "clip",
                clip_fraction=0.051,
            )
        )

    assert edge["eligible"] is False
    assert "max_edge_violation_above_0.0005_m" in edge[
        "eligibility_failures"
    ]


def test_candidate_rejects_declared_contraction_that_robot_does_not_have(
    tmp_path: Path,
) -> None:
    root = _write_candidate_fixture(
        tmp_path / "mismatch",
        declared_contraction=0.1,
        actual_contraction=0.0,
    )

    with pytest.raises(ValueError, match="declared contraction"):
        candidate_record(root)


def test_candidate_rejects_report_from_different_control_motion(
    tmp_path: Path,
) -> None:
    root = _write_candidate_fixture(tmp_path / "mismatched-control")
    desired_path = root / "desired.json"
    desired = json.loads(desired_path.read_text(encoding="utf-8"))
    desired["frames"][0]["keypoints"]["spine_front"][0] += 0.06
    _write_json(desired_path, desired)

    with pytest.raises(ValueError, match="status counts differ"):
        candidate_record(root)


def test_candidate_rejects_forged_status_counts_and_fraction(
    tmp_path: Path,
) -> None:
    root = _write_candidate_fixture(tmp_path / "forged-status")
    report_path = root / "reachability_report.json"
    run_summary_path = root / "run_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["frames"][0]["status"] = "marginal"
    report["summary"]["status_counts"] = {
        "feasible": 1,
        "marginal": 1,
        "unreachable": 0,
    }
    report["summary"]["feasible_fraction"] = 0.5
    _write_json(report_path, report)
    _write_json(run_summary_path, {"summary": report["summary"]})

    with pytest.raises(ValueError, match="status counts differ"):
        candidate_record(root)


def test_comparison_report_uses_case_relative_artifact_paths(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case"
    candidate = _write_candidate_fixture(case / "candidates" / "candidate")

    report = compare_candidates([candidate])

    selected = report["selected_candidate"]
    assert selected["candidate_root"] == "candidates/candidate"
    assert all(
        not Path(item["path"]).is_absolute()
        for item in selected["artifacts"].values()
    )


def _build_complete_case(root: Path) -> None:
    for folder in (
        "observation",
        "initial_model",
        "landmarks",
        "control",
        "motion",
        "previews",
        "reports",
        "sources",
        "candidates/selected",
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)

    video_path = root / "sources" / "video.mp4"
    predictions_path = root / "sources" / "predictions.h5"
    mesh_path = root / "sources" / "mesh.glb"
    corrected_path = root / "sources" / "corrected.json"
    _write_video(video_path)
    columns = pd.MultiIndex.from_product(
        [
            ["fixture"],
            ["animal0"],
            SUPERANIMAL_QUADRUPED_39,
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    prediction_rows = np.empty((272, len(columns)), dtype=float)
    for role_idx, _role in enumerate(SUPERANIMAL_QUADRUPED_39):
        prediction_rows[:, role_idx * 3 : role_idx * 3 + 3] = (
            float(4 + role_idx % 24),
            float(3 + role_idx % 15),
            0.9,
        )
    back_base_index = SUPERANIMAL_QUADRUPED_39.index("back_base")
    prediction_rows[0, back_base_index * 3] += 0.1
    pd.DataFrame(prediction_rows, columns=columns).to_hdf(
        predictions_path,
        key="predictions",
        mode="w",
    )
    mesh_path.write_bytes(b"glb")
    video_sha = _sha(video_path)
    predictions_sha = _sha(predictions_path)
    video = {
        "width": 32,
        "height": 24,
        "fps": 30.0,
        "frame_count": 272,
    }

    corrected_positions = {
        "spine_front": (24.0, 10.0),
        "spine_rear": (8.0, 10.0),
        "front_left_foot": (24.0, 19.0),
        "front_right_foot": (22.0, 19.0),
        "rear_left_foot": (10.0, 19.0),
        "rear_right_foot": (8.0, 19.0),
    }
    corrected = {
        "schema": "qianji.keypoint_trajectory_2d",
        "schema_version": "1.3.0",
        "coordinate_system": (
            "image_pixels_top_left_origin_x_right_y_down"
        ),
        "video": video,
        "identity_anchor": {
            "frame_idx": 152,
            "timestamp_s": 152 / 30.0,
            "front_assignment": "keep",
            "rear_assignment": "keep",
            "confirmed_by": "manual",
            "video_sha256": video_sha,
            "predictions_sha256": predictions_sha,
        },
        "source": {
            "video_sha256": video_sha,
            "predictions_sha256": predictions_sha,
        },
        "frames": [
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / 30.0,
                "keypoints": {
                    role: {
                        "x_px": corrected_positions[role][0],
                        "y_px": corrected_positions[role][1],
                        "confidence": 0.9,
                        "valid": True,
                        "flags": [],
                    }
                    for role in KEYPOINT_ROLES
                },
            }
            for frame_idx in range(272)
        ],
    }
    _write_json(corrected_path, corrected)
    corrected_sha = _sha(corrected_path)
    source_paths = {
        "video": video_path,
        "predictions": predictions_path,
        "mesh": mesh_path,
        "corrected_spine": corrected_path,
    }
    _write_json(
        root / "input_manifest.json",
        {
            "schema": "qianji.cat_39point_input_manifest",
            "schema_version": "0.1.0",
            "reference_frame": 152,
            "identity_anchor": {
                "frame_idx": 152,
                "front_assignment": "keep",
                "rear_assignment": "keep",
                "confirmed_by": "manual",
            },
            "sources": {
                key: {"path": str(path), "sha256": _sha(path)}
                for key, path in source_paths.items()
            },
            "video": video,
            "predictions": {
                "scorer": "fixture",
                "individual": "animal0",
                "bodyparts": list(SUPERANIMAL_QUADRUPED_39),
                "coordinates": ["x", "y", "likelihood"],
            },
            "mesh_provenance": {
                "generation_method": "hunyuan3d_from_video_frame",
                "provenance_basis": "fixture",
                "verified_conversion_metadata": "fixture",
            },
            "scientific_limits": {
                "reconstruction_kind": "body_relative_2_5d_retarget",
                "metric_depth_observed": False,
                "camera_calibrated": False,
                "global_translation_preserved": False,
                "triangle_mesh_deformed": False,
                "mesh_vertices_deformed": False,
                "mesh_reconstructed_in_pipeline": False,
                "dynamics_simulated": False,
            },
        },
    )

    observation_result = build_39point_observation(
        pd.read_hdf(predictions_path),
        VideoInfo(width=32, height=24, fps=30.0, frame_count=272),
        individual="animal0",
        confidence_threshold=0.3,
        anchor_frame=152,
        front_anchor="keep",
        rear_anchor="keep",
    )
    observation = observation_result.trajectory
    source_lineage = {
        "video_sha256": video_sha,
        "predictions_sha256": predictions_sha,
        "corrected_trajectory_sha256": corrected_sha,
    }
    observation["source_lineage"] = source_lineage
    observation_result.report["source_lineage"] = dict(source_lineage)
    _write_json(
        root / "observation" / "keypoint_trajectory_2d_39.json",
        observation,
    )
    _write_json(
        root / "observation" / "keypoint_39_quality_report.json",
        observation_result.report,
    )
    _write_video(root / "observation" / "keypoint_39_preview.mp4")

    robot = _robot()
    neutral_positions = np.asarray(
        [item["pos"] for item in robot["sites"].values()],
        dtype=float,
    )
    site_names = list(robot["sites"])
    for rod in robot["rod_groups"]:
        site1 = neutral_positions[site_names.index(rod["site1"])]
        site2 = neutral_positions[site_names.index(rod["site2"])]
        current = float(np.linalg.norm(site2 - site1))
        rod["constraint"] = {
            "mode": "variable_length_fixed_stretch_ratio",
            "effective_current_length": current,
            "effective_min_length": current,
            "effective_max_length": current + 1.0,
            "slide_control_mode": "relative_around_initial",
            "slide_min_each_side_required": 0.0,
            "slide_max_each_side_required": 0.5,
            "slide_range_each_side_required": 0.5,
            "permitted_contraction_fraction": 0.0,
        }
        rod["actuated"] = True
    robot["metadata"] = {"permitted_contraction_fraction": 0.0}
    rig = {
        "schema": "qianji-key-site-map",
        "key_site_map": {
            role: f"s{index:03d}"
            for index, role in enumerate(KEYPOINT_ROLES)
        },
    }
    _write_json(root / "initial_model" / "robot_base.json", robot)
    _write_json(
        root / "initial_model" / "base_robot_manifest.json",
        {
            "schema": "qianji.initial_vgt_model",
            "schema_version": "1.0.0",
            "source_mesh": {
                "path": str(mesh_path),
                "sha256": _sha(mesh_path),
            },
            "template": {
                "sha256": _sha(root / "initial_model" / "robot_base.json"),
            },
            "qianji_generation": {
                "repository_commit": "3" * 40,
                "entrypoint": "morph_generator/generate.py",
                "parameters": {"preset": "abstract", "seed": 4},
            },
            "artifact": {
                "path": "initial_model/robot_base.json",
                "sha256": _sha(root / "initial_model" / "robot_base.json"),
                "sites": 12,
                "rods": 30,
            },
            "frozen_precomputed_from_mesh": True,
            "reason": "fixture",
        },
    )
    _write_json(root / "initial_model" / "robot.json", robot)
    _write_json(root / "initial_model" / "rig_bbox.json", rig)
    _write_json(root / "initial_model" / "rig_keypoints.json", rig)
    anchor_xml = "".join(
        (
            f'<body name="anchor_{name}" '
            f'pos="{" ".join(str(value) for value in item["pos"])}"/>'
        )
        for name, item in robot["sites"].items()
    )
    joint_xml = "".join(
        (
            f'<joint name="{rod["name"]}__slide_{side}" '
            f'type="slide" axis="0 0 {"-1" if side == "left" else "1"}" '
            'range="0.0 0.5"/>'
        )
        for rod in robot["rod_groups"]
        for side in ("left", "right")
    )
    endpoint_site_xml = "".join(
        (
            f'<site name="{rod["name"]}__s1_site"/>'
            f'<site name="{rod["name"]}__s2_site"/>'
            f'<site name="anchor_{rod["site1"]}_port_{rod["name"]}_left_site"/>'
            f'<site name="anchor_{rod["site2"]}_port_{rod["name"]}_right_site"/>'
        )
        for rod in robot["rod_groups"]
    )
    weld_xml = "".join(
        (
            f'<weld site1="{rod["name"]}__s1_site" '
            f'site2="anchor_{rod["site1"]}_port_{rod["name"]}_left_site"/>'
            f'<weld site1="{rod["name"]}__s2_site" '
            f'site2="anchor_{rod["site2"]}_port_{rod["name"]}_right_site"/>'
        )
        for rod in robot["rod_groups"]
    )
    actuator_xml = "".join(
        (
            f'<position name="act_{rod["name"]}__slide_{side}" '
            f'joint="{rod["name"]}__slide_{side}" ctrlrange="0.0 0.5"/>'
        )
        for rod in robot["rod_groups"]
        for side in ("left", "right")
    )
    (root / "initial_model" / "robot.xml").write_text(
        (
            '<mujoco model="selected"><worldbody>'
            f"{anchor_xml}{joint_xml}{endpoint_site_xml}</worldbody>"
            f"<equality>{weld_xml}</equality>"
            f"<actuator>{actuator_xml}</actuator></mujoco>\n"
        ),
        encoding="utf-8",
    )
    (root / "initial_model" / "robot_scene.xml").write_text(
        (
            '<mujoco model="selected"><worldbody>'
            f"{anchor_xml}{joint_xml}{endpoint_site_xml}</worldbody>"
            f"<equality>{weld_xml}</equality>"
            f"<actuator>{actuator_xml}</actuator></mujoco>\n"
        ),
        encoding="utf-8",
    )

    neutral = build_neutral_landmarks_39(
        observation,
        corrected,
        robot,
        rig,
        reference_frame=152,
    )
    lifted = lift_39point_trajectory(
        observation,
        corrected,
        neutral,
        motion_scale=0.1,
    )
    lift_sources = {
        "trajectory_39": {
            "path": "observation/keypoint_trajectory_2d_39.json",
            "sha256": _sha(
                root / "observation" / "keypoint_trajectory_2d_39.json"
            ),
        },
        "corrected_spine": {
            "path": str(corrected_path),
            "sha256": corrected_sha,
        },
        "robot": {
            "path": "initial_model/robot_base.json",
            "sha256": _sha(root / "initial_model" / "robot_base.json"),
        },
        "rig": {
            "path": "initial_model/rig_bbox.json",
            "sha256": _sha(root / "initial_model" / "rig_bbox.json"),
        },
    }
    neutral["sources"] = lift_sources
    lifted.report["sources"] = lift_sources
    _write_json(root / "landmarks" / "neutral_landmarks_39.json", neutral)
    _write_json(
        root / "landmarks" / "keypoint_motion_3d_39.json",
        lifted.motion,
    )
    _write_json(root / "landmarks" / "lift_39_report.json", lifted.report)

    control_map = build_vgt_control_map(robot, rig, neutral)
    _write_json(root / "control" / "vgt_control_map.json", control_map)
    desired_payload, _ = build_target_control_motion(
        lifted.motion,
        neutral,
        robot,
        control_map,
    )
    projected_payload = json.loads(json.dumps(desired_payload))
    projected_payload["frames"][0]["keypoints"]["spine_front"][:3] = (
        robot["sites"][rig["key_site_map"]["spine_front"]]["pos"]
    )
    desired = root / "control" / "target_control_keypoint_motion.json"
    projected = root / "control" / "projected_control_keypoint_motion.json"
    _write_json(desired, desired_payload)
    _write_json(projected, projected_payload)

    npz_path = root / "motion" / "vgt_motion.npz"
    times = np.arange(272, dtype=float) / 30.0
    positions = np.repeat(neutral_positions[None, :, :], 272, axis=0)
    np.savez_compressed(
        npz_path,
        site_names=np.asarray(site_names),
        times=times,
        positions=positions,
    )
    _write_video(root / "previews" / "vgt_motion_30fps.mp4")
    image = np.full((100, 100, 3), 245, dtype=np.uint8)
    cv2.line(image, (5, 5), (95, 95), (10, 50, 100), 3)
    cv2.imwrite(str(root / "previews" / "vgt_three_view.png"), image)
    cv2.imwrite(str(root / "previews" / "vgt_isometric.png"), image)

    selected_root = root / "candidates" / "selected"
    sequence = load_and_validate_vgt_sequence(
        npz_path,
        robot,
        expected_frames=272,
        expected_sites=12,
        expected_rods=30,
    )
    desired_motion = validate_control_motion(
        desired_payload,
        expected_frames=272,
        expected_fps=30.0,
    )
    projected_motion = validate_control_motion(
        projected_payload,
        expected_frames=272,
        expected_fps=30.0,
    )
    independent = recompute_reachability_metrics(
        desired_motion,
        projected_motion,
        sequence,
        robot,
    )
    summary = independent["summary"]
    run_summary = selected_root / "run_summary.json"
    qianji_report = selected_root / "reachability_report.json"
    _write_json(run_summary, {"summary": summary})
    _write_json(
        qianji_report,
        {
            "schema": "qianji-keypoint-reachability-report-v1",
            "extension_only": True,
            "thresholds": REACHABILITY_THRESHOLDS,
            "summary": summary,
            "frames": [
                {
                    **independent["frames"][index],
                    "target_keypoints": {
                        role: [*value[:3], 1.0]
                        for role, value in desired_payload["frames"][index][
                            "keypoints"
                        ].items()
                    },
                    "projected_keypoints": projected_payload["frames"][
                        index
                    ]["keypoints"],
                }
                for index in range(272)
            ],
        },
    )
    geometry = verify_module.compute_rod_constraint_metrics(sequence, robot)

    def artifact(path: Path) -> dict:
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha(path),
        }

    selected = {
        "candidate_id": "selected",
        "candidate_root": "candidates/selected",
        "motion_scale": 0.1,
        "contraction_fraction": 0.0,
        "actual_contraction_fraction": 0.0,
        "extension_only": True,
        "rig_variant": "bbox",
        "morphology_variant": "base",
        "summary": summary,
        "recomputed_reachability": {
            "thresholds": REACHABILITY_THRESHOLDS,
            "summary": summary,
        },
        "recomputed_geometry": geometry,
        "eligible": True,
        "eligibility_failures": [],
        "artifacts": {
            "run_summary": artifact(run_summary),
            "reachability_report": artifact(qianji_report),
            "robot": artifact(root / "initial_model" / "robot.json"),
            "rig": artifact(root / "initial_model" / "rig_keypoints.json"),
            "desired_motion": artifact(desired),
            "projected_motion": artifact(projected),
            "site_npz": artifact(npz_path),
        },
    }
    _write_json(root / "reports" / "selected_candidate.json", selected)
    acceptance = {
        "unreachable_frames": 0,
        "max_keypoint_error_m": 0.05,
        "max_edge_violation_m": 0.0005,
        "max_estimated_clipped_fraction": 0.05,
        "max_recomputed_violated_rod_fraction": 0.05,
        "sites": 12,
        "rods": 30,
    }
    _write_json(
        root / "reports" / "reachability_report.json",
        {
            "schema": "qianji.39point_reachability_comparison",
            "schema_version": "0.1.0",
            "acceptance": acceptance,
            "candidates": [selected],
            "comparisons": {},
            "selected_candidate": selected,
            "selection_rationale": "fixture",
        },
    )

    conversion_paths = {
        "source_robot": root / "initial_model" / "robot.json",
        "robot_xml": root / "initial_model" / "robot.xml",
        "robot_scene_xml": root / "initial_model" / "robot_scene.xml",
    }
    _write_json(
        root / "initial_model" / "robot_conversion_manifest.json",
        {
            "schema": "qianji.selected_robot_conversion",
            "schema_version": "1.0.0",
            "converter": (
                "QianJi mujoco_builder/json2xml_v7_perrod.py"
            ),
            "artifacts": {
                label: artifact(path)
                for label, path in conversion_paths.items()
            },
        },
    )

    npz_artifact = artifact(npz_path)
    _write_json(
        root / "motion" / "vgt_motion_manifest.json",
        {
            "schema": "qianji.vgt_motion_manifest",
            "schema_version": "0.1.0",
            "selected_candidate": selected,
            "frame_count": 272,
            "fps": 30.0,
            "time_range": [0.0, 271 / 30.0],
            "site_count": 12,
            "site_names": site_names,
            "rod_count": 30,
            "rods": [
                {
                    "name": item["name"],
                    "site1": item["site1"],
                    "site2": item["site2"],
                }
                for item in robot["rod_groups"]
            ],
            "times_shape": [272],
            "positions_shape": [272, 12, 3],
            "vgt_motion": {
                **npz_artifact,
                "source_path": npz_artifact["path"],
                "source_sha256": npz_artifact["sha256"],
            },
            "robot": artifact(root / "initial_model" / "robot.json"),
            "rig": {
                **artifact(root / "initial_model" / "rig_keypoints.json"),
                "key_site_map": rig["key_site_map"],
            },
            "desired_control_motion": {
                **artifact(desired),
                "schema": "qianji-keypoint-trajectory-v1",
            },
            "projected_control_motion": {
                **artifact(projected),
                "schema": "qianji-keypoint-trajectory-v1",
            },
            "scientific_limits": {
                "metric_depth_observed": False,
                "camera_calibrated": False,
                "global_translation_preserved": False,
                "dynamics_simulated": False,
                "mesh_vertices_deformed": False,
            },
        },
    )
    _write_json(
        root / "reports" / "experiment_provenance.json",
        {
            "schema": "qianji.experiment_provenance",
            "schema_version": "1.0.0",
            "repository": {
                "path": str(ROOT),
                "commit": "1" * 40,
                "dirty": False,
                "status_porcelain": [],
                "worktree_state_sha256": "2" * 64,
            },
            "qianji": {
                "path": str(ROOT),
                "commit": "3" * 40,
                "dirty": False,
                "status_porcelain": [],
                "worktree_state_sha256": "4" * 64,
            },
            "scripts": [
                {
                    "path": str(ROOT / f"script-{index}.py"),
                    "sha256": f"{index + 1:064x}",
                }
                for index in range(10)
            ],
            "inputs": [
                {"path": str(path), "sha256": _sha(path)}
                for path in (
                    video_path,
                    predictions_path,
                    mesh_path,
                    corrected_path,
                    ROOT
                    / "experiments/39point_vgt_cat/fixtures/"
                    "cat_hunyuan_qianji_robot_12x30.json",
                    ROOT
                    / "experiments/39point_vgt_cat/fixtures/"
                    "cat_hunyuan_qianji_robot_12x30.manifest.json",
                )
            ],
            "invocation": {
                "argv": [
                    "bash",
                    str(
                        ROOT
                        / "experiments/39point_vgt_cat/run_experiment.sh"
                    ),
                ],
                "working_directory": str(ROOT),
            },
            "environment": {
                "python_executable": "/fixture/python",
                "python_version": "3.12.0",
                "packages": {
                    "mujoco": "fixture",
                    "numpy": "fixture",
                    "pandas": "fixture",
                    "scipy": "fixture",
                },
            },
            "qianji_environment": {
                "python_executable": "/fixture/qianji/python",
                "python_version": "3.10.0",
                "packages": {
                    "mujoco": "fixture",
                    "numpy": "fixture",
                    "pandas": None,
                    "scipy": "fixture",
                },
            },
        },
    )


def test_full_case_verifier_names_each_failed_acceptance_gate(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case"
    _build_complete_case(case)
    assert verify_case(case)["passed"] is True

    trajectory_path = case / "observation" / "keypoint_trajectory_2d_39.json"
    trajectory = json.loads(trajectory_path.read_text())
    removed = trajectory["frames"][0]["keypoints"].pop("nose")
    _write_json(trajectory_path, trajectory)
    report = verify_case(case)
    assert report["passed"] is False
    assert report["checks"]["observation_39_roles"]["passed"] is False
    trajectory["frames"][0]["keypoints"]["nose"] = removed
    trajectory["frames"][0]["keypoints"] = {
        role: trajectory["frames"][0]["keypoints"][role]
        for role in SUPERANIMAL_QUADRUPED_39
    }
    _write_json(trajectory_path, trajectory)

    robot_path = case / "initial_model" / "robot.json"
    robot = json.loads(robot_path.read_text())
    rod = robot["rod_groups"].pop()
    _write_json(robot_path, robot)
    report = verify_case(case)
    assert report["checks"]["robot_12_sites_30_rods"]["passed"] is False
    robot["rod_groups"].append(rod)
    _write_json(robot_path, robot)

    manifest_path = case / "motion" / "vgt_motion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    saved_hash = manifest["vgt_motion"]["sha256"]
    manifest["vgt_motion"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    report = verify_case(case)
    assert report["checks"]["manifest_hashes"]["passed"] is False
    manifest["vgt_motion"]["sha256"] = saved_hash
    _write_json(manifest_path, manifest)

    saved_projected = dict(manifest["projected_control_motion"])
    manifest["projected_control_motion"] = dict(
        manifest["desired_control_motion"]
    )
    _write_json(manifest_path, manifest)
    report = verify_case(case)
    assert report["checks"]["desired_projected_distinct"]["passed"] is False
    manifest["projected_control_motion"] = saved_projected
    _write_json(manifest_path, manifest)

    saved_limit = manifest["scientific_limits"].pop("dynamics_simulated")
    _write_json(manifest_path, manifest)
    report = verify_case(case)
    assert report["checks"]["scientific_limits"]["passed"] is False
    manifest["scientific_limits"]["dynamics_simulated"] = saved_limit
    _write_json(manifest_path, manifest)

    preview = case / "previews" / "vgt_isometric.png"
    backup = preview.read_bytes()
    preview.unlink()
    report = verify_case(case)
    assert report["checks"]["required_files"]["passed"] is False
    preview.write_bytes(backup)


@pytest.mark.parametrize(
    ("mutation", "check_name"),
    [
        ("invalid_with_coordinates", "observation_39_roles"),
        ("confidence_below_zero", "observation_39_roles"),
        ("confidence_above_one", "observation_39_roles"),
        ("antler_applicable", "body_relative_3d_39"),
        ("missing_interpolation_field", "body_relative_3d_39"),
        ("reversed_3d_time", "body_relative_3d_39"),
        ("substitution_mismatch", "body_relative_3d_39"),
        ("lift_source_hash", "body_relative_3d_39"),
        ("missing_control_site", "observation_control_split"),
    ],
)
def test_verifier_rejects_each_semantic_contract_violation(
    tmp_path: Path,
    mutation: str,
    check_name: str,
) -> None:
    case = tmp_path / mutation
    _build_complete_case(case)
    if mutation.startswith(("invalid_", "confidence_")):
        path = case / "observation" / "keypoint_trajectory_2d_39.json"
        payload = json.loads(path.read_text())
        point = payload["frames"][0]["keypoints"]["nose"]
        if mutation == "invalid_with_coordinates":
            point["valid"] = False
            point["flags"] = ["low_confidence"]
        elif mutation == "confidence_below_zero":
            point["confidence"] = -0.1
        else:
            point["confidence"] = 1.1
    elif mutation == "antler_applicable":
        path = case / "landmarks" / "neutral_landmarks_39.json"
        payload = json.loads(path.read_text())
        payload["landmarks"]["right_antler_base"][
            "anatomy_applicable"
        ] = True
    elif mutation == "missing_control_site":
        path = case / "control" / "vgt_control_map.json"
        payload = json.loads(path.read_text())
        payload["controls"]["spine_front"]["site"] = "missing"
    elif mutation == "lift_source_hash":
        path = case / "landmarks" / "neutral_landmarks_39.json"
        payload = json.loads(path.read_text())
        payload["sources"]["trajectory_39"]["sha256"] = "0" * 64
    else:
        path = case / "landmarks" / "lift_39_report.json"
        if mutation == "reversed_3d_time":
            path = case / "landmarks" / "keypoint_motion_3d_39.json"
        payload = json.loads(path.read_text())
        if mutation == "missing_interpolation_field":
            payload.pop("interpolation_applied")
        elif mutation == "reversed_3d_time":
            payload["frames"][1]["time"] = 0.0
        else:
            payload["substitutions"].pop()
    _write_json(path, payload)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"][check_name]["passed"] is False


def _publish_selected_record(case: Path, selected: dict) -> None:
    _write_json(case / "reports" / "selected_candidate.json", selected)
    reachability_path = case / "reports" / "reachability_report.json"
    reachability = json.loads(reachability_path.read_text())
    reachability["selected_candidate"] = selected
    reachability["candidates"] = [selected]
    _write_json(reachability_path, reachability)
    manifest_path = case / "motion" / "vgt_motion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_candidate"] = selected
    _write_json(manifest_path, manifest)


def test_verifier_rejects_projected_controls_that_do_not_match_npz(
    tmp_path: Path,
) -> None:
    case = tmp_path / "projected_mismatch"
    _build_complete_case(case)
    projected_path = (
        case / "control" / "projected_control_keypoint_motion.json"
    )
    projected = json.loads(projected_path.read_text())
    projected["frames"][1]["keypoints"]["spine_front"][0] += 0.01
    _write_json(projected_path, projected)
    digest = _sha(projected_path)

    selected = json.loads(
        (case / "reports" / "selected_candidate.json").read_text()
    )
    selected["artifacts"]["projected_motion"]["sha256"] = digest
    _publish_selected_record(case, selected)
    manifest_path = case / "motion" / "vgt_motion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["projected_control_motion"]["sha256"] = digest
    _write_json(manifest_path, manifest)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"]["manifest_hashes"]["passed"] is False
    assert "does not match VGT NPZ" in report["checks"]["manifest_hashes"]["error"]


def test_verifier_recomputes_rod_limits_from_npz(
    tmp_path: Path,
) -> None:
    case = tmp_path / "rod_violation"
    _build_complete_case(case)
    npz_path = case / "motion" / "vgt_motion.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        site_names = np.array(archive["site_names"], copy=True)
        times = np.array(archive["times"], copy=True)
        positions = np.array(archive["positions"], copy=True)
    positions[1, 6, 0] += 10.0
    np.savez_compressed(
        npz_path,
        site_names=site_names,
        times=times,
        positions=positions,
    )
    digest = _sha(npz_path)

    selected = json.loads(
        (case / "reports" / "selected_candidate.json").read_text()
    )
    selected["artifacts"]["site_npz"]["sha256"] = digest
    _publish_selected_record(case, selected)
    manifest_path = case / "motion" / "vgt_motion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["vgt_motion"]["sha256"] = digest
    manifest["vgt_motion"]["source_sha256"] = digest
    _write_json(manifest_path, manifest)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"]["reachability_accepted"]["passed"] is False


def test_verifier_rejects_canonical_robot_hash_mismatch(
    tmp_path: Path,
) -> None:
    case = tmp_path / "canonical_robot"
    _build_complete_case(case)
    robot_path = case / "initial_model" / "robot.json"
    robot = json.loads(robot_path.read_text())
    robot["name"] = "not-the-selected-robot"
    _write_json(robot_path, robot)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"]["robot_12_sites_30_rods"]["passed"] is False


def test_verifier_rejects_base_robot_that_is_not_linked_to_the_input_mesh(
    tmp_path: Path,
) -> None:
    case = tmp_path / "base_lineage"
    _build_complete_case(case)
    manifest_path = case / "initial_model" / "base_robot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_mesh"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"]["initial_vgt_lineage"]["passed"] is False


def test_verifier_replays_h5_instead_of_trusting_observation_lineage(
    tmp_path: Path,
) -> None:
    case = tmp_path / "h5-replay"
    _build_complete_case(case)
    predictions_path = case / "sources" / "predictions.h5"
    dataframe = pd.read_hdf(predictions_path)
    x_column = (
        "fixture",
        "animal0",
        "nose",
        "x",
    )
    dataframe.loc[0, x_column] += 1.0
    dataframe.to_hdf(predictions_path, key="predictions", mode="w")
    predictions_sha = _sha(predictions_path)

    corrected_path = case / "sources" / "corrected.json"
    corrected = json.loads(corrected_path.read_text())
    corrected["source"]["predictions_sha256"] = predictions_sha
    corrected["identity_anchor"]["predictions_sha256"] = predictions_sha
    _write_json(corrected_path, corrected)
    corrected_sha = _sha(corrected_path)

    manifest_path = case / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"]["predictions"]["sha256"] = predictions_sha
    manifest["sources"]["corrected_spine"]["sha256"] = corrected_sha
    _write_json(manifest_path, manifest)

    lineage = {
        "video_sha256": manifest["sources"]["video"]["sha256"],
        "predictions_sha256": predictions_sha,
        "corrected_trajectory_sha256": corrected_sha,
    }
    trajectory_path = (
        case / "observation" / "keypoint_trajectory_2d_39.json"
    )
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["source_lineage"] = lineage
    _write_json(trajectory_path, trajectory)
    quality_path = case / "observation" / "keypoint_39_quality_report.json"
    quality = json.loads(quality_path.read_text())
    quality["source_lineage"] = lineage
    _write_json(quality_path, quality)

    lift_sources = {
        "trajectory_39": {
            "path": "observation/keypoint_trajectory_2d_39.json",
            "sha256": _sha(trajectory_path),
        },
        "corrected_spine": {
            "path": str(corrected_path),
            "sha256": corrected_sha,
        },
    }
    neutral_path = case / "landmarks" / "neutral_landmarks_39.json"
    neutral = json.loads(neutral_path.read_text())
    neutral["sources"].update(lift_sources)
    _write_json(neutral_path, neutral)
    lift_report_path = case / "landmarks" / "lift_39_report.json"
    lift_report = json.loads(lift_report_path.read_text())
    lift_report["sources"].update(lift_sources)
    _write_json(lift_report_path, lift_report)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"]["observation_39_roles"]["passed"] is False
    assert "independent replay" in report["checks"][
        "observation_39_roles"
    ]["error"]


@pytest.mark.parametrize(
    ("mutation", "check_name"),
    [
        ("valid_3d_xyz", "body_relative_3d_39"),
        ("lateral_3d_xyz", "body_relative_3d_39"),
        ("neutral_primary_xyz", "observation_control_split"),
        ("desired_control_xyz", "observation_control_split"),
    ],
)
def test_verifier_replays_each_upstream_transform(
    tmp_path: Path,
    mutation: str,
    check_name: str,
) -> None:
    case = tmp_path / mutation
    _build_complete_case(case)
    if mutation in {"valid_3d_xyz", "lateral_3d_xyz"}:
        path = case / "landmarks" / "keypoint_motion_3d_39.json"
        payload = json.loads(path.read_text())
        payload["frames"][0]["keypoints"]["nose"][
            0 if mutation == "valid_3d_xyz" else 1
        ] += 0.01
    elif mutation == "neutral_primary_xyz":
        path = case / "control" / "vgt_control_map.json"
        payload = json.loads(path.read_text())
        payload["controls"]["spine_front"]["neutral_primary_xyz"][0] += 0.01
    else:
        path = case / "control" / "target_control_keypoint_motion.json"
        payload = json.loads(path.read_text())
        payload["frames"][0]["keypoints"]["spine_front"][0] += 0.01
    _write_json(path, payload)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"][check_name]["passed"] is False
    assert "independent replay" in report["checks"][check_name]["error"]


def test_verifier_rejects_xml_generated_from_a_different_robot(
    tmp_path: Path,
) -> None:
    case = tmp_path / "canonical_xml"
    _build_complete_case(case)
    xml_path = case / "initial_model" / "robot.xml"
    xml = xml_path.read_text()
    xml_path.write_text(
        xml.replace('name="anchor_s000" pos="0.5 0.0 0.8"', (
            'name="anchor_s000" pos="9.0 0.0 0.8"'
        )),
        encoding="utf-8",
    )
    conversion_path = (
        case / "initial_model" / "robot_conversion_manifest.json"
    )
    conversion = json.loads(conversion_path.read_text())
    conversion["artifacts"]["robot_xml"]["sha256"] = _sha(xml_path)
    _write_json(conversion_path, conversion)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"]["robot_12_sites_30_rods"]["passed"] is False
    assert "XML anchors differ" in report["checks"][
        "robot_12_sites_30_rods"
    ]["error"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("slide_axis", "axis differs"),
        ("weld_endpoint", "weld endpoints differ"),
        ("actuator_joint", "targets the wrong joint"),
    ],
)
def test_verifier_checks_xml_rod_topology(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    case = tmp_path / mutation
    _build_complete_case(case)
    xml_path = case / "initial_model" / "robot.xml"
    tree = ET.parse(xml_path)
    xml_root = tree.getroot()
    if mutation == "slide_axis":
        next(
            joint
            for joint in xml_root.iter("joint")
            if joint.get("name", "").endswith("__slide_left")
        ).set("axis", "0 0 1")
    elif mutation == "weld_endpoint":
        welds = list(xml_root.iter("weld"))
        welds[0].set("site2", str(welds[1].get("site2")))
    else:
        actuators = list(xml_root.iter("position"))
        actuators[0].set("joint", str(actuators[1].get("joint")))
    tree.write(xml_path, encoding="unicode")
    conversion_path = (
        case / "initial_model" / "robot_conversion_manifest.json"
    )
    conversion = json.loads(conversion_path.read_text())
    conversion["artifacts"]["robot_xml"]["sha256"] = _sha(xml_path)
    _write_json(conversion_path, conversion)

    report = verify_case(case)

    assert report["passed"] is False
    check = report["checks"]["robot_12_sites_30_rods"]
    assert check["passed"] is False
    assert message in check["error"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_invocation",
        "missing_required_package",
        "missing_qianji_environment",
        "missing_qianji_package",
        "malformed_script_hash",
    ],
)
def test_verifier_rejects_incomplete_experiment_provenance(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = tmp_path / mutation
    _build_complete_case(case)
    provenance_path = case / "reports" / "experiment_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if mutation == "missing_invocation":
        provenance.pop("invocation")
    elif mutation == "missing_required_package":
        provenance["environment"]["packages"].pop("scipy")
    elif mutation == "missing_qianji_environment":
        provenance.pop("qianji_environment")
    elif mutation == "missing_qianji_package":
        provenance["qianji_environment"]["packages"].pop("numpy")
    else:
        provenance["scripts"][0]["sha256"] = "not-a-sha256"
    _write_json(provenance_path, provenance)

    report = verify_case(case)

    assert report["passed"] is False
    assert report["checks"]["experiment_provenance"]["passed"] is False
