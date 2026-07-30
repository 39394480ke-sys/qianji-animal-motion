from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qianji_animal_motion.keypoints_39 import SUPERANIMAL_QUADRUPED_39


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


def _write_video(path: Path, *, frames: int = 272, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), 245, dtype=np.uint8)
        frame[:, index % 32] = (20, 80, 180)
        writer.write(frame)
    writer.release()


def _robot() -> dict:
    sites = {
        f"s{index:03d}": {
            "pos": [float(index % 4), float(index // 4), float(index % 3)]
        }
        for index in range(12)
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
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
    source_paths = {}
    for name, content in (
        ("video.mp4", b"input-video"),
        ("predictions.h5", b"39-point-h5"),
        ("mesh.glb", b"glb"),
        ("corrected.json", b"corrected"),
    ):
        path = root / "sources" / name
        path.write_bytes(content)
        source_paths[name] = path
    _write_json(
        root / "input_manifest.json",
        {
            "schema": "qianji.input_manifest",
            "video": {"width": 32, "height": 24, "fps": 30.0, "frame_count": 272},
            "sources": {
                key: {"path": str(path), "sha256": _sha(path)}
                for key, path in source_paths.items()
            },
            "mesh_generation_method": "hunyuan3d_from_video_frame",
            "scientific_limits": {
                "mesh_reconstructed_in_pipeline": False,
                "metric_depth_observed": False,
                "dynamics_simulated": False,
                "mesh_vertices_deformed": False,
            },
        },
    )
    observation_frames = []
    motion_frames = []
    for frame_idx in range(272):
        observation_frames.append(
            {
                "frame_idx": frame_idx,
                "keypoints": {
                    role: {
                        "raw_x_px": 10.0,
                        "raw_y_px": 10.0,
                        "x_px": 10.0,
                        "y_px": 10.0,
                        "confidence": 0.9,
                        "valid": True,
                    }
                    for role in SUPERANIMAL_QUADRUPED_39
                },
            }
        )
        motion_frames.append(
            {
                "time": frame_idx / 30.0,
                "keypoints": {
                    role: [0.1, 0.2, 0.3, 0.9]
                    for role in SUPERANIMAL_QUADRUPED_39
                },
            }
        )
    _write_json(
        root / "observation" / "keypoint_trajectory_2d_39.json",
        {
            "schema": "qianji.keypoint_trajectory_2d_39",
            "roles": list(SUPERANIMAL_QUADRUPED_39),
            "video": {"width": 32, "height": 24, "fps": 30.0, "frame_count": 272},
            "frames": observation_frames,
        },
    )
    _write_json(
        root / "observation" / "keypoint_39_quality_report.json",
        {
            "schema": "qianji.keypoint_39_quality_report",
            "frame_count": 272,
            "role_count": 39,
            "temporal_interpolation": False,
            "missing_source_values": 0,
        },
    )
    _write_video(root / "observation" / "keypoint_39_preview.mp4")
    robot = _robot()
    _write_json(root / "initial_model" / "robot.json", robot)
    rig = {
        "schema": "qianji-key-site-map",
        "key_site_map": {
            role: f"s{index:03d}"
            for index, role in enumerate(
                (
                    "spine_front",
                    "spine_rear",
                    "front_left_foot",
                    "front_right_foot",
                    "rear_left_foot",
                    "rear_right_foot",
                )
            )
        },
    }
    _write_json(root / "initial_model" / "rig_keypoints.json", rig)
    _write_json(
        root / "landmarks" / "neutral_landmarks_39.json",
        {
            "schema": "qianji.neutral_landmarks_39",
            "reference_frame": 152,
            "landmarks": {
                role: {"xyz": [0.1, 0.2, 0.3], "anatomy_applicable": True}
                for role in SUPERANIMAL_QUADRUPED_39
            },
            "scientific_limits": {"metric_depth_observed": False},
        },
    )
    _write_json(
        root / "landmarks" / "keypoint_motion_3d_39.json",
        {
            "schema": "qianji-keypoint-trajectory-39-v1",
            "reconstruction_kind": "body_relative_2_5d_retarget",
            "fps": 30.0,
            "reference_frame": 152,
            "frames": motion_frames,
        },
    )
    _write_json(
        root / "landmarks" / "lift_39_report.json",
        {
            "schema": "qianji.keypoint_lift_39_report",
            "frame_count": 272,
            "role_count": 39,
            "temporal_interpolation": False,
            "reconstruction_kind": "body_relative_2_5d_retarget",
        },
    )
    controls = {}
    primary = (
        "back_base",
        "back_end",
        "front_left_paw",
        "front_right_paw",
        "back_left_paw",
        "back_right_paw",
    )
    for index, role in enumerate(rig["key_site_map"]):
        controls[role] = {
            "site": f"s{index:03d}",
            "observation_primary": primary[index],
            "observation_fallback": None,
        }
    _write_json(
        root / "control" / "vgt_control_map.json",
        {
            "schema": "qianji.vgt_control_map",
            "observation_role_count": 39,
            "control_role_count": 6,
            "controls": controls,
        },
    )
    desired = root / "control" / "target_control_keypoint_motion.json"
    projected = root / "control" / "projected_control_keypoint_motion.json"
    _write_json(desired, {"schema": "desired", "frames": motion_frames})
    projected_frames = json.loads(json.dumps(motion_frames))
    projected_frames[1]["keypoints"]["nose"][0] += 0.001
    _write_json(projected, {"schema": "projected", "frames": projected_frames})
    site_names = np.asarray(list(robot["sites"]))
    times = np.arange(272, dtype=float) / 30.0
    positions = np.repeat(
        np.asarray([item["pos"] for item in robot["sites"].values()])[None, :, :],
        272,
        axis=0,
    )
    positions[:, 0, 2] += 0.01 * np.sin(np.arange(272) / 10.0)
    np.savez_compressed(
        root / "motion" / "vgt_motion.npz",
        site_names=site_names,
        times=times,
        positions=positions,
    )
    _write_video(root / "previews" / "vgt_motion_30fps.mp4")
    image = np.full((100, 100, 3), 245, dtype=np.uint8)
    cv2.line(image, (5, 5), (95, 95), (10, 50, 100), 3)
    cv2.imwrite(str(root / "previews" / "vgt_three_view.png"), image)
    cv2.imwrite(str(root / "previews" / "vgt_isometric.png"), image)
    vgt_npz = root / "motion" / "vgt_motion.npz"
    _write_json(
        root / "motion" / "vgt_motion_manifest.json",
        {
            "schema": "qianji.vgt_motion_manifest",
            "frame_count": 272,
            "fps": 30.0,
            "site_count": 12,
            "rod_count": 30,
            "site_names": list(robot["sites"]),
            "positions_shape": [272, 12, 3],
            "vgt_motion": {"path": str(vgt_npz), "sha256": _sha(vgt_npz)},
            "robot": {
                "path": str(root / "initial_model" / "robot.json"),
                "sha256": _sha(root / "initial_model" / "robot.json"),
            },
            "rig": {
                "path": str(root / "initial_model" / "rig_keypoints.json"),
                "sha256": _sha(root / "initial_model" / "rig_keypoints.json"),
            },
            "desired_control_motion": {
                "path": str(desired),
                "sha256": _sha(desired),
            },
            "projected_control_motion": {
                "path": str(projected),
                "sha256": _sha(projected),
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
        root / "reports" / "reachability_report.json",
        {
            "schema": "qianji.39point_reachability_comparison",
            "selected_candidate": {
                "candidate_id": "selected",
                "eligible": True,
                "motion_scale": 0.1,
                "summary": {
                    "frames": 272,
                    "status_counts": {
                        "feasible": 10,
                        "marginal": 262,
                        "unreachable": 0,
                    },
                    "feasible_fraction": 10 / 272,
                    "max_keypoint_error_m": 0.04,
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
