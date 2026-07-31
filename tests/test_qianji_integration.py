from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
PINNED_QIANJI_COMMIT = "3f3676c6cb7c198f7c9c43ce0b002f61d3d524a8"
PINNED_FILE_HASHES = {
    "mujoco_builder/json2xml_v7_perrod.py": (
        "77ebb0857042b48556abff65f54c7e51c7beec94cc468cf2ba8c2a655c4f77d4"
    ),
    "mujoco_builder/robot_config.py": (
        "176699944ecfd0d1f597e86eafcf45d7cbc28f35213e53c8579a9058eee9cf19"
    ),
    "LICENSE": (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_qianji_converter_preserves_slide_and_weld_contract(
    tmp_path: Path,
) -> None:
    qianji_value = os.environ.get("QIANJI_INTEGRATION_ROOT")
    if qianji_value is None:
        pytest.skip("real QianJi integration requires its private checkout")
    qianji_root = Path(qianji_value).resolve()
    commit = subprocess.run(
        ["git", "-C", str(qianji_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == PINNED_QIANJI_COMMIT
    for relative, expected_hash in PINNED_FILE_HASHES.items():
        assert _sha256(qianji_root / relative) == expected_hash

    robot_path = (
        ROOT
        / "experiments/39point_vgt_cat/fixtures/"
        "cat_hunyuan_qianji_robot_12x30.json"
    )
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    xml_path = tmp_path / "robot.xml"
    subprocess.run(
        [
            sys.executable,
            str(qianji_root / "mujoco_builder/json2xml_v7_perrod.py"),
            "-i",
            str(robot_path),
            "-o",
            str(xml_path),
            "--model-name",
            "qianji_ci_contract",
        ],
        check=True,
        cwd=qianji_root,
    )

    xml_root = ET.parse(xml_path).getroot()
    joints = {
        joint.get("name"): joint
        for joint in xml_root.iter("joint")
        if joint.get("name")
    }
    actuators = {
        actuator.get("name"): actuator
        for actuator in xml_root.iter("position")
        if actuator.get("name")
    }
    welds = {
        (weld.get("site1"), weld.get("site2"))
        for weld in xml_root.iter("weld")
    }
    expected_welds = set()
    for rod in robot["rod_groups"]:
        name = rod["name"]
        constraint = rod["constraint"]
        slide_range = constraint["slide_range_each_side_required"]
        expected_range = np.asarray(
            [
                constraint.get("slide_min_each_side_required", 0.0),
                constraint.get("slide_max_each_side_required", slide_range),
            ],
            dtype=float,
        )
        for side, expected_axis in (
            ("left", [0.0, 0.0, -1.0]),
            ("right", [0.0, 0.0, 1.0]),
        ):
            joint_name = f"{name}__slide_{side}"
            joint = joints[joint_name]
            assert joint.get("type") == "slide"
            np.testing.assert_allclose(
                np.fromstring(joint.get("axis", ""), sep=" "),
                expected_axis,
            )
            np.testing.assert_allclose(
                np.fromstring(joint.get("range", ""), sep=" "),
                expected_range,
            )
            actuator = actuators[f"act_{joint_name}"]
            assert actuator.get("joint") == joint_name
            np.testing.assert_allclose(
                np.fromstring(actuator.get("ctrlrange", ""), sep=" "),
                expected_range,
            )
        expected_welds.update(
            {
                (
                    f"{name}__s1_site",
                    f"anchor_{rod['site1']}_port_{name}_left_site",
                ),
                (
                    f"{name}__s2_site",
                    f"anchor_{rod['site2']}_port_{name}_right_site",
                ),
            }
        )
    assert welds == expected_welds
