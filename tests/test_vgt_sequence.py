from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qianji_animal_motion.vgt_render import render_vgt_motion
from qianji_animal_motion.vgt_sequence import (
    VgtSequence,
    compute_rod_constraint_metrics,
    load_and_validate_vgt_sequence,
    validate_control_motion,
    validate_control_pair_against_sequence,
)
from qianji_animal_motion.vgt_sequence_cli import run_package_vgt_motion


def _robot() -> dict:
    sites = {}
    for index in range(12):
        angle = 2.0 * np.pi * index / 12.0
        sites[f"s{index:03d}"] = {
            "pos": [
                float(np.cos(angle)),
                float(np.sin(angle)),
                float(0.2 + 0.1 * (index % 4)),
            ]
        }
    names = list(sites)
    rods = []
    pairs = []
    for offset in (1, 2, 3):
        for index in range(12):
            pair = tuple(sorted((names[index], names[(index + offset) % 12])))
            if pair not in pairs:
                pairs.append(pair)
    for index, (site1, site2) in enumerate(pairs[:30]):
        rods.append(
            {
                "name": f"r{index:03d}",
                "site1": site1,
                "site2": site2,
                "constraint": {
                    "effective_current_length": 1.0,
                    "effective_min_length": 0.9,
                    "effective_max_length": 1.5,
                },
            }
        )
    return {"name": "cat", "sites": sites, "rod_groups": rods}


def _rig() -> dict:
    return {
        "schema": "qianji-key-site-map",
        "key_site_map": {
            "spine_front": "s000",
            "spine_rear": "s006",
            "front_left_foot": "s001",
            "front_right_foot": "s011",
            "rear_left_foot": "s005",
            "rear_right_foot": "s007",
        },
    }


def _arrays(frames: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = _robot()
    site_names = np.asarray(list(robot["sites"]))
    times = np.arange(frames, dtype=float) / 30.0
    neutral = np.asarray([item["pos"] for item in robot["sites"].values()])
    positions = np.repeat(neutral[None, :, :], frames, axis=0)
    positions[:, 0, 2] += np.arange(frames) * 0.15
    return site_names, times, positions


def _write_npz(path: Path, *, frames: int = 3) -> None:
    site_names, times, positions = _arrays(frames)
    np.savez_compressed(
        path,
        site_names=site_names,
        times=times,
        positions=positions,
    )


def _control_motion(*, frames: int = 3, desired: bool = False) -> dict:
    robot = _robot()
    rig = _rig()
    site_names, times, positions = _arrays(frames)
    indices = {name: index for index, name in enumerate(site_names.tolist())}
    output_frames = []
    for frame_idx in range(frames):
        keypoints = {}
        for role, site in rig["key_site_map"].items():
            xyz = positions[frame_idx, indices[site]].copy()
            if desired and role == "spine_front":
                xyz[0] += 0.01
            keypoints[role] = [*xyz.astype(float).tolist(), 0.9]
        output_frames.append(
            {"time": float(times[frame_idx]), "keypoints": keypoints}
        )
    return {
        "schema": "qianji-keypoint-trajectory-v1",
        "fps": 30.0,
        "source": "fixture",
        "frames": output_frames,
    }


def test_loads_exact_site_time_position_and_rod_contract(tmp_path: Path) -> None:
    path = tmp_path / "source.npz"
    _write_npz(path)

    sequence = load_and_validate_vgt_sequence(
        path,
        _robot(),
        expected_frames=3,
        expected_sites=12,
        expected_rods=30,
    )

    assert sequence.site_names == tuple(_robot()["sites"])
    assert sequence.times.shape == (3,)
    assert sequence.positions.shape == (3, 12, 3)
    assert len(sequence.rods) == 30
    assert all(site1 in sequence.site_names for site1, _ in sequence.rods)
    assert all(site2 in sequence.site_names for _, site2 in sequence.rods)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nan", "finite"),
        ("duplicate_site", "unique"),
        ("wrong_frames", "frames"),
        ("wrong_sites", "sites"),
        ("nonmonotonic_time", "increasing"),
    ],
)
def test_rejects_invalid_npz_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    site_names, times, positions = _arrays()
    if mutation == "nan":
        positions[1, 2, 0] = np.nan
    elif mutation == "duplicate_site":
        site_names[1] = site_names[0]
    elif mutation == "wrong_frames":
        positions = positions[:2]
    elif mutation == "wrong_sites":
        positions = positions[:, :11]
        site_names = site_names[:11]
    elif mutation == "nonmonotonic_time":
        times[2] = times[1]
    path = tmp_path / f"{mutation}.npz"
    np.savez(path, site_names=site_names, times=times, positions=positions)

    with pytest.raises(ValueError, match=message):
        load_and_validate_vgt_sequence(
            path,
            _robot(),
            expected_frames=3,
            expected_sites=12,
            expected_rods=30,
        )


def test_rejects_missing_rod_endpoint_and_wrong_rod_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.npz"
    _write_npz(path)
    missing = _robot()
    missing["rod_groups"][0]["site2"] = "missing"
    with pytest.raises(ValueError, match="missing site"):
        load_and_validate_vgt_sequence(
            path,
            missing,
            expected_frames=3,
            expected_sites=12,
            expected_rods=30,
        )
    wrong_count = _robot()
    wrong_count["rod_groups"].pop()
    with pytest.raises(ValueError, match="30 rods"):
        load_and_validate_vgt_sequence(
            path,
            wrong_count,
            expected_frames=3,
            expected_sites=12,
            expected_rods=30,
        )


def test_direct_rod_constraint_metrics_detect_tampered_site_position() -> None:
    site_names, times, positions = _arrays()
    robot = _robot()
    for rod in robot["rod_groups"]:
        rod["constraint"]["effective_min_length"] = 0.0
        rod["constraint"]["effective_max_length"] = 1000.0
    first = robot["rod_groups"][0]
    site_indices = {
        name: index for index, name in enumerate(site_names.tolist())
    }
    reference_length = float(
        np.linalg.norm(
            positions[0, site_indices[first["site2"]]]
            - positions[0, site_indices[first["site1"]]]
        )
    )
    first["constraint"]["effective_max_length"] = reference_length + 0.001
    positions[1, site_indices[first["site2"]]] += [10.0, 0.0, 0.0]
    sequence = VgtSequence(
        site_names=tuple(site_names.tolist()),
        times=times,
        positions=positions,
        rods=tuple(
            (rod["site1"], rod["site2"]) for rod in robot["rod_groups"]
        ),
    )

    metrics = compute_rod_constraint_metrics(sequence, robot)

    assert metrics["max_edge_violation_m"] > 1.0
    assert metrics["max_violated_rod_fraction"] >= 1.0 / 30.0
    assert metrics["frame_max_edge_violation_m"][0] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda motion: motion.update({"schema": "wrong"}),
            "schema",
        ),
        (
            lambda motion: motion["frames"][1].update({"time": 0.0}),
            "increasing",
        ),
        (
            lambda motion: motion["frames"][0]["keypoints"].pop(
                "rear_right_foot"
            ),
            "six control roles",
        ),
        (
            lambda motion: motion["frames"][0]["keypoints"][
                "spine_front"
            ].__setitem__(3, 1.1),
            "confidence",
        ),
    ],
)
def test_control_motion_validator_rejects_semantic_errors(
    mutation,
    message: str,
) -> None:
    motion = _control_motion()
    mutation(motion)

    with pytest.raises(ValueError, match=message):
        validate_control_motion(
            motion,
            expected_frames=3,
            expected_fps=30.0,
        )


def test_control_motion_accepts_exact_roles_in_json_independent_order() -> None:
    motion = _control_motion()
    for frame in motion["frames"]:
        frame["keypoints"] = dict(
            sorted(frame["keypoints"].items())
        )

    validated = validate_control_motion(
        motion,
        expected_frames=3,
        expected_fps=30.0,
    )

    assert validated.positions.shape == (3, 6, 3)


def test_control_pair_accepts_exact_rig_roles_in_json_independent_order() -> None:
    site_names, times, positions = _arrays()
    robot = _robot()
    sequence = VgtSequence(
        site_names=tuple(site_names.tolist()),
        times=times,
        positions=positions,
        rods=tuple(
            (rod["site1"], rod["site2"]) for rod in robot["rod_groups"]
        ),
    )
    rig = _rig()
    rig["key_site_map"] = dict(sorted(rig["key_site_map"].items()))

    desired, projected = validate_control_pair_against_sequence(
        _control_motion(desired=True),
        _control_motion(),
        sequence,
        rig,
        expected_frames=3,
        expected_fps=30.0,
    )

    assert desired.positions.shape == projected.positions.shape == (3, 6, 3)


def test_renderer_outputs_visible_four_view_30fps_motion(tmp_path: Path) -> None:
    site_names, times, positions = _arrays(frames=2)
    robot = _robot()
    sequence = VgtSequence(
        site_names=tuple(site_names.tolist()),
        times=times,
        positions=positions,
        rods=tuple(
            (item["site1"], item["site2"])
            for item in robot["rod_groups"]
        ),
    )

    report = render_vgt_motion(
        sequence,
        robot,
        tmp_path,
        width=1246,
        height=720,
        fps=30.0,
    )

    three_view = cv2.imread(str(tmp_path / "vgt_three_view.png"))
    isometric = cv2.imread(str(tmp_path / "vgt_isometric.png"))
    assert three_view is not None and np.any(three_view < 245)
    assert isometric is not None and np.any(isometric < 245)
    capture = cv2.VideoCapture(str(tmp_path / "vgt_motion_30fps.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1246
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 720
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0, abs=0.01)
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert len(decoded) == 2
    assert np.mean(cv2.absdiff(decoded[0], decoded[1])) > 0.01
    height, width = decoded[0].shape[:2]
    panels = (
        decoded[0][: height // 2, : width // 2],
        decoded[0][: height // 2, width // 2 :],
        decoded[0][height // 2 :, : width // 2],
        decoded[0][height // 2 :, width // 2 :],
    )
    assert all(np.any(panel < 245) for panel in panels)
    assert report["frame_count"] == 2
    assert report["rod_count"] == 30


def test_package_cli_copies_validated_sequence_and_hashes_sources(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "feasible_site_targets.npz"
    _write_npz(source_npz)
    sources = {
        "robot": _robot(),
        "rig": _rig(),
        "desired": _control_motion(desired=True),
        "projected": _control_motion(),
    }
    paths = {}
    for label, payload in sources.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[label] = path
    selected = {
        "candidate_id": "scale_010_bbox_base_c010",
        "eligible": True,
        "artifacts": {
            "robot": {
                "sha256": hashlib.sha256(paths["robot"].read_bytes()).hexdigest()
            },
            "rig": {
                "sha256": hashlib.sha256(paths["rig"].read_bytes()).hexdigest()
            },
            "desired_motion": {
                "sha256": hashlib.sha256(
                    paths["desired"].read_bytes()
                ).hexdigest()
            },
            "projected_motion": {
                "sha256": hashlib.sha256(
                    paths["projected"].read_bytes()
                ).hexdigest()
            },
            "site_npz": {
                "sha256": hashlib.sha256(source_npz.read_bytes()).hexdigest()
            },
        },
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    paths["selected"] = selected_path
    output = tmp_path / "final"

    outputs = run_package_vgt_motion(
        source_npz_path=source_npz,
        robot_path=paths["robot"],
        rig_path=paths["rig"],
        desired_motion_path=paths["desired"],
        projected_motion_path=paths["projected"],
        selected_candidate_path=paths["selected"],
        output_dir=output,
        expected_frames=3,
        expected_sites=12,
        expected_rods=30,
        width=1246,
        height=720,
        fps=30.0,
    )

    assert len(outputs) == 5
    assert all(path.is_file() for path in outputs)
    packaged = np.load(output / "vgt_motion.npz", allow_pickle=False)
    assert packaged["positions"].shape == (3, 12, 3)
    manifest = json.loads(
        (output / "vgt_motion_manifest.json").read_text()
    )
    assert manifest["selected_candidate"]["candidate_id"] == (
        "scale_010_bbox_base_c010"
    )
    assert manifest["positions_shape"] == [3, 12, 3]
    assert manifest["site_count"] == 12
    assert manifest["rod_count"] == 30
    assert manifest["fps"] == 30.0
    assert manifest["desired_control_motion"]["sha256"] != (
        manifest["projected_control_motion"]["sha256"]
    )
    assert manifest["vgt_motion"]["sha256"] == hashlib.sha256(
        (output / "vgt_motion.npz").read_bytes()
    ).hexdigest()
    assert manifest["scientific_limits"]["metric_depth_observed"] is False
    for record_name in (
        "vgt_motion",
        "robot",
        "rig",
        "desired_control_motion",
        "projected_control_motion",
    ):
        assert not Path(manifest[record_name]["path"]).is_absolute()

    alias = tmp_path / "alias"
    with pytest.raises(ValueError, match="different files"):
        run_package_vgt_motion(
            source_npz_path=source_npz,
            robot_path=paths["robot"],
            rig_path=paths["rig"],
            desired_motion_path=paths["desired"],
            projected_motion_path=paths["desired"],
            selected_candidate_path=paths["selected"],
            output_dir=alias,
            expected_frames=3,
            expected_sites=12,
            expected_rods=30,
            width=1246,
            height=720,
            fps=30.0,
        )


def test_package_rejects_projected_motion_not_matching_npz(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "feasible_site_targets.npz"
    _write_npz(source_npz)
    payloads = {
        "robot": _robot(),
        "rig": _rig(),
        "desired": _control_motion(desired=True),
        "projected": _control_motion(),
    }
    payloads["projected"]["frames"][1]["keypoints"]["spine_front"][0] += 0.2
    paths = {}
    for label, payload in payloads.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[label] = path
    selected = {
        "candidate_id": "candidate",
        "eligible": True,
        "artifacts": {
            "robot": {"sha256": hashlib.sha256(paths["robot"].read_bytes()).hexdigest()},
            "rig": {"sha256": hashlib.sha256(paths["rig"].read_bytes()).hexdigest()},
            "desired_motion": {
                "sha256": hashlib.sha256(paths["desired"].read_bytes()).hexdigest()
            },
            "projected_motion": {
                "sha256": hashlib.sha256(paths["projected"].read_bytes()).hexdigest()
            },
            "site_npz": {"sha256": hashlib.sha256(source_npz.read_bytes()).hexdigest()},
        },
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    with pytest.raises(ValueError, match="projected control motion"):
        run_package_vgt_motion(
            source_npz_path=source_npz,
            robot_path=paths["robot"],
            rig_path=paths["rig"],
            desired_motion_path=paths["desired"],
            projected_motion_path=paths["projected"],
            selected_candidate_path=selected_path,
            output_dir=tmp_path / "final",
            expected_frames=3,
            expected_sites=12,
            expected_rods=30,
            fps=30.0,
        )


def test_package_rejects_selected_candidate_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "feasible_site_targets.npz"
    _write_npz(source_npz)
    paths = {}
    for label, payload in {
        "robot": _robot(),
        "rig": _rig(),
        "desired": _control_motion(desired=True),
        "projected": _control_motion(),
    }.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[label] = path
    selected = {
        "candidate_id": "candidate",
        "eligible": True,
        "artifacts": {
            "robot": {"sha256": "0" * 64},
            "rig": {"sha256": hashlib.sha256(paths["rig"].read_bytes()).hexdigest()},
            "desired_motion": {
                "sha256": hashlib.sha256(paths["desired"].read_bytes()).hexdigest()
            },
            "projected_motion": {
                "sha256": hashlib.sha256(paths["projected"].read_bytes()).hexdigest()
            },
            "site_npz": {"sha256": hashlib.sha256(source_npz.read_bytes()).hexdigest()},
        },
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    with pytest.raises(ValueError, match="selected candidate robot hash"):
        run_package_vgt_motion(
            source_npz_path=source_npz,
            robot_path=paths["robot"],
            rig_path=paths["rig"],
            desired_motion_path=paths["desired"],
            projected_motion_path=paths["projected"],
            selected_candidate_path=selected_path,
            output_dir=tmp_path / "final",
            expected_frames=3,
            expected_sites=12,
            expected_rods=30,
            fps=30.0,
        )
