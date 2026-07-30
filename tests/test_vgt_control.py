from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from qianji_animal_motion.lift_3d import KEYPOINT_ROLES
from qianji_animal_motion.vgt_control import (
    apply_contraction_range,
    build_motion_informed_rig,
    build_target_control_motion,
    build_vgt_control_map,
    infer_uniform_contraction_fraction,
    validate_vgt_control_map,
)
from qianji_animal_motion.vgt_control_cli import run_prepare_vgt_control


CONTROL_OBSERVATIONS = {
    "spine_front": ("back_base", "neck_base"),
    "spine_rear": ("back_end", "tail_base"),
    "front_left_foot": ("front_left_paw", None),
    "front_right_foot": ("front_right_paw", None),
    "rear_left_foot": ("back_left_paw", None),
    "rear_right_foot": ("back_right_paw", None),
}


def _robot() -> dict:
    sites = {
        "s00": {"pos": [-1.0, 0.0, 0.8]},
        "s01": {"pos": [1.0, 0.0, 0.8]},
        "s02": {"pos": [1.0, 0.4, 0.0]},
        "s03": {"pos": [1.0, -0.4, 0.0]},
        "s04": {"pos": [-1.0, 0.4, 0.0]},
        "s05": {"pos": [-1.0, -0.4, 0.0]},
    }
    for index in range(6, 12):
        sites[f"s{index:02d}"] = {
            "pos": [-0.5 + 0.2 * (index - 6), 0.0, 0.4]
        }
    rods = []
    site_names = list(sites)
    for index in range(30):
        site1 = site_names[index % 12]
        site2 = site_names[(index * 5 + 1) % 12]
        if site1 == site2:
            site2 = site_names[(index + 1) % 12]
        rods.append(
            {
                "name": f"r{index:02d}",
                "site1": site1,
                "site2": site2,
                "constraint": {
                    "mode": "variable_length_fixed_stretch_ratio",
                    "effective_current_length": 1.0 + 0.01 * index,
                    "effective_min_length": 1.0 + 0.01 * index,
                    "effective_max_length": 1.5 + 0.01 * index,
                },
            }
        )
    return {"name": "cat", "sites": sites, "rod_groups": rods}


def _bbox_rig() -> dict:
    return {
        "schema": "qianji-key-site-map",
        "version": "0.1",
        "source": "test_bbox",
        "key_site_map": {
            "spine_front": "s01",
            "spine_rear": "s00",
            "front_left_foot": "s02",
            "front_right_foot": "s03",
            "rear_left_foot": "s04",
            "rear_right_foot": "s05",
        },
    }


def _neutral() -> dict:
    values = {
        "back_base": [0.8, 0.0, 0.8],
        "neck_base": [0.7, 0.0, 0.8],
        "back_end": [-0.8, 0.0, 0.8],
        "tail_base": [-0.7, 0.0, 0.8],
        "front_left_paw": [0.9, 0.35, 0.0],
        "front_right_paw": [0.9, -0.35, 0.0],
        "back_left_paw": [-0.9, 0.35, 0.0],
        "back_right_paw": [-0.9, -0.35, 0.0],
    }
    return {
        "schema": "qianji.neutral_landmarks_39",
        "reference_frame": 0,
        "landmarks": {
            role: {
                "xyz": xyz,
                "anatomy_applicable": True,
            }
            for role, xyz in values.items()
        },
    }


def _motion() -> dict:
    neutral = _neutral()["landmarks"]
    reference = {
        role: [*item["xyz"], 0.9]
        for role, item in neutral.items()
    }
    moved = copy.deepcopy(reference)
    moved["back_base"] = [9.0, 9.0, 9.0, 0.0]
    moved["neck_base"] = [0.7, 0.0, 1.0, 0.7]
    moved["front_left_paw"] = [1.0, 0.35, 0.0, 0.8]
    return {
        "schema": "qianji-keypoint-trajectory-39-v1",
        "fps": 30.0,
        "reference_frame": 0,
        "frames": [
            {"time": 0.0, "keypoints": reference},
            {"time": 1.0 / 30.0, "keypoints": moved},
        ],
    }


def test_control_map_separates_observations_from_six_distinct_sites() -> None:
    robot = _robot()
    rig = _bbox_rig()
    neutral = _neutral()

    control_map = build_vgt_control_map(robot, rig, neutral)
    target, report = build_target_control_motion(
        _motion(),
        neutral,
        robot,
        control_map,
    )

    assert tuple(control_map["controls"]) == KEYPOINT_ROLES
    assert len({item["site"] for item in control_map["controls"].values()}) == 6
    for role, (primary, fallback) in CONTROL_OBSERVATIONS.items():
        assert control_map["controls"][role]["observation_primary"] == primary
        assert control_map["controls"][role]["observation_fallback"] == fallback
        np.testing.assert_allclose(
            target["frames"][0]["keypoints"][role][:3],
            robot["sites"][rig["key_site_map"][role]]["pos"],
        )

    # back_base is invalid, so only its declared same-frame neck_base fallback
    # may drive the structural spine_front target.
    np.testing.assert_allclose(
        target["frames"][1]["keypoints"]["spine_front"],
        [1.0, 0.0, 1.0, 0.7],
    )
    assert report["fallbacks"] == [
        {
            "frame_idx": 1,
            "control_role": "spine_front",
            "invalid_primary": "back_base",
            "used_observation": "neck_base",
        }
    ]
    assert report["neutral_substitutions"] == []
    assert target["schema"] == "qianji-keypoint-trajectory-v1"
    validate_vgt_control_map(control_map, robot, rig)

    invalid = copy.deepcopy(control_map)
    invalid["controls"]["spine_front"]["site"] = "missing"
    with pytest.raises(ValueError, match="missing"):
        validate_vgt_control_map(invalid, robot, rig)

    invalid_transfer = copy.deepcopy(control_map)
    invalid_transfer["controls"]["spine_front"]["transfer"] = "unknown"
    with pytest.raises(ValueError, match="transfer"):
        validate_vgt_control_map(invalid_transfer, robot, rig)


def test_missing_primary_without_declared_fallback_uses_neutral() -> None:
    motion = _motion()
    motion["frames"][1]["keypoints"]["back_right_paw"][-1] = 0.0
    target, report = build_target_control_motion(
        motion,
        _neutral(),
        _robot(),
        build_vgt_control_map(_robot(), _bbox_rig(), _neutral()),
    )

    np.testing.assert_allclose(
        target["frames"][1]["keypoints"]["rear_right_foot"],
        [-1.0, -0.4, 0.0, 0.0],
    )
    assert report["neutral_substitutions"] == [
        {
            "frame_idx": 1,
            "control_role": "rear_right_foot",
            "reason": "no_valid_observation",
        }
    ]


def test_motion_informed_rig_uses_minimum_distinct_assignment() -> None:
    rig = build_motion_informed_rig(_robot(), _neutral())

    assert rig["key_site_map"] == {
        "spine_front": "s01",
        "spine_rear": "s00",
        "front_left_foot": "s02",
        "front_right_foot": "s03",
        "rear_left_foot": "s04",
        "rear_right_foot": "s05",
    }
    assert rig["source"] == "motion_informed_neutral_landmarks"
    assert rig["assignment"]["distinct_sites"] is True

    robot = _robot()
    robot["sites"] = dict(list(robot["sites"].items())[:5])
    with pytest.raises(ValueError, match="at least six"):
        build_motion_informed_rig(robot, _neutral())


def test_contraction_range_uses_qianji_recognized_bidirectional_fields() -> None:
    robot = _robot()
    original = copy.deepcopy(robot)

    contracted = apply_contraction_range(robot, 0.10)

    assert robot == original
    for before, after in zip(
        robot["rod_groups"],
        contracted["rod_groups"],
        strict=True,
    ):
        before_constraint = before["constraint"]
        after_constraint = after["constraint"]
        assert after_constraint["effective_min_length"] == pytest.approx(
            0.9 * before_constraint["effective_current_length"]
        )
        assert after_constraint["effective_current_length"] == (
            before_constraint["effective_current_length"]
        )
        assert after_constraint["effective_max_length"] == (
            before_constraint["effective_max_length"]
        )
        assert after_constraint["mode"] == before_constraint["mode"]
        assert after_constraint["slide_control_mode"] == (
            "relative_around_initial"
        )
        assert after_constraint["slide_min_each_side_required"] == (
            pytest.approx(
                -0.05 * before_constraint["effective_current_length"]
            )
        )
        assert after_constraint["slide_max_each_side_required"] == (
            pytest.approx(
                0.5
                * (
                    before_constraint["effective_max_length"]
                    - before_constraint["effective_current_length"]
                )
            )
        )
        assert after_constraint["slide_range_each_side_required"] == (
            after_constraint["slide_max_each_side_required"]
        )
        assert after_constraint["permitted_contraction_fraction"] == 0.10
    assert contracted["metadata"]["permitted_contraction_fraction"] == 0.10
    assert infer_uniform_contraction_fraction(contracted) == pytest.approx(0.10)

    for fraction in (-0.01, 0.500001, float("nan")):
        with pytest.raises(ValueError, match="fraction"):
            apply_contraction_range(robot, fraction)


def test_infer_contraction_rejects_nonuniform_or_unrecognized_robot() -> None:
    robot = apply_contraction_range(_robot(), 0.10)
    robot["rod_groups"][0]["constraint"]["effective_min_length"] = 0.95
    robot["rod_groups"][0]["constraint"][
        "slide_min_each_side_required"
    ] = -0.025
    with pytest.raises(ValueError, match="uniform"):
        infer_uniform_contraction_fraction(robot)

    robot = apply_contraction_range(_robot(), 0.10)
    del robot["rod_groups"][0]["constraint"]["slide_control_mode"]
    with pytest.raises(ValueError, match="slide_control_mode"):
        infer_uniform_contraction_fraction(robot)


def test_prepare_cli_publishes_versioned_candidate_inputs(tmp_path: Path) -> None:
    source_values = {
        "robot": _robot(),
        "bbox_rig": _bbox_rig(),
        "neutral": _neutral(),
        "motion": _motion(),
    }
    source_paths = {}
    for name, payload in source_values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        source_paths[name] = path

    output = tmp_path / "control"
    paths = run_prepare_vgt_control(
        robot_path=source_paths["robot"],
        bbox_rig_path=source_paths["bbox_rig"],
        neutral_landmarks_path=source_paths["neutral"],
        motion_39_path=source_paths["motion"],
        output_dir=output,
        contraction_fractions=(0.0, 0.1),
    )

    assert len(paths) == 11
    assert all(path.is_file() for path in paths)
    assert (output / "rig_bbox.json").is_file()
    assert (output / "rig_motion_informed.json").is_file()
    assert (output / "robot_contraction_000.json").is_file()
    assert (output / "robot_contraction_010.json").is_file()
    manifest = json.loads(
        (output / "vgt_control_candidates_manifest.json").read_text()
    )
    assert manifest["source_hashes"].keys() == source_values.keys()
    assert manifest["contraction_fractions"] == [0.0, 0.1]
    with pytest.raises(FileExistsError):
        run_prepare_vgt_control(
            robot_path=source_paths["robot"],
            bbox_rig_path=source_paths["bbox_rig"],
            neutral_landmarks_path=source_paths["neutral"],
            motion_39_path=source_paths["motion"],
            output_dir=output,
            contraction_fractions=(0.0, 0.1),
        )
