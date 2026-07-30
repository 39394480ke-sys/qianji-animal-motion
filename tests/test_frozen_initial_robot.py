from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qianji_animal_motion.frozen_initial_robot import (
    prepare_frozen_initial_robot,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    mesh = tmp_path / "source.glb"
    mesh.write_bytes(b"fixed mesh")
    robot = tmp_path / "robot_template.json"
    _write_json(
        robot,
        {
            "name": "frozen-vgt",
            "sites": {
                "s0": {"pos": [0.0, 0.0, 0.0]},
                "s1": {"pos": [1.0, 0.0, 0.0]},
            },
            "rod_groups": [
                {
                    "name": "r0",
                    "site1": "s0",
                    "site2": "s1",
                    "actuated": True,
                }
            ],
        },
    )
    manifest = tmp_path / "template_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "qianji.frozen_mesh_vgt_template",
            "schema_version": "1.0.0",
            "source_mesh": {"sha256": _sha256(mesh)},
            "template": {
                "sha256": _sha256(robot),
                "sites": 2,
                "rods": 1,
            },
            "qianji_generation": {
                "repository_commit": "a" * 40,
                "entrypoint": "morph_generator/generate.py",
                "parameters": {"preset": "abstract", "seed": 4},
            },
        },
    )
    return mesh, robot, manifest


def test_prepare_frozen_initial_robot_publishes_verified_copy_and_lineage(
    tmp_path: Path,
) -> None:
    mesh, robot, manifest = _fixture(tmp_path)
    output = tmp_path / "case" / "initial_model"

    result = prepare_frozen_initial_robot(
        mesh_path=mesh,
        template_path=robot,
        template_manifest_path=manifest,
        output_dir=output,
    )

    copied = output / "robot_base.json"
    lineage = json.loads(
        (output / "base_robot_manifest.json").read_text(encoding="utf-8")
    )
    assert copied.read_bytes() == robot.read_bytes()
    assert result == lineage
    assert lineage["source_mesh"]["sha256"] == _sha256(mesh)
    assert lineage["artifact"] == {
        "path": "initial_model/robot_base.json",
        "sha256": _sha256(copied),
        "sites": 2,
        "rods": 1,
    }
    assert lineage["frozen_precomputed_from_mesh"] is True


@pytest.mark.parametrize(
    "mutation",
    ["mesh_hash", "template_hash", "topology"],
)
def test_prepare_frozen_initial_robot_rejects_unverified_inputs_atomically(
    tmp_path: Path,
    mutation: str,
) -> None:
    mesh, robot, manifest = _fixture(tmp_path)
    if mutation == "mesh_hash":
        mesh.write_bytes(b"different mesh")
    elif mutation == "template_hash":
        robot.write_text("{}\n", encoding="utf-8")
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["template"]["sites"] = 12
        _write_json(manifest, payload)
    output = tmp_path / "case" / "initial_model"

    with pytest.raises(ValueError):
        prepare_frozen_initial_robot(
            mesh_path=mesh,
            template_path=robot,
            template_manifest_path=manifest,
            output_dir=output,
        )

    assert not (output / "robot_base.json").exists()
    assert not (output / "base_robot_manifest.json").exists()


def test_checked_in_cat_template_matches_its_frozen_manifest() -> None:
    fixture_root = (
        REPOSITORY_ROOT / "experiments" / "39point_vgt_cat" / "fixtures"
    )
    robot_path = fixture_root / "cat_hunyuan_qianji_robot_12x30.json"
    manifest = json.loads(
        (
            fixture_root
            / "cat_hunyuan_qianji_robot_12x30.manifest.json"
        ).read_text(encoding="utf-8")
    )
    robot = json.loads(robot_path.read_text(encoding="utf-8"))

    assert _sha256(robot_path) == manifest["template"]["sha256"]
    assert len(robot["sites"]) == manifest["template"]["sites"] == 12
    assert len(robot["rod_groups"]) == manifest["template"]["rods"] == 30
