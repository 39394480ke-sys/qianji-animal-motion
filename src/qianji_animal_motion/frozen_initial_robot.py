"""Publish a verified, frozen QianJi VGT model for a known source Mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from qianji_animal_motion.artifact_io import publish_staged_files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_object(path: Path) -> dict:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _validate_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_robot_topology(
    robot: dict,
    *,
    expected_sites: int,
    expected_rods: int,
) -> None:
    sites = robot.get("sites")
    rods = robot.get("rod_groups")
    if not isinstance(sites, dict) or len(sites) != expected_sites:
        raise ValueError(
            f"frozen robot must contain exactly {expected_sites} sites"
        )
    if not isinstance(rods, list) or len(rods) != expected_rods:
        raise ValueError(
            f"frozen robot must contain exactly {expected_rods} rods"
        )
    names = set(sites)
    if len(names) != len(sites) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("frozen robot site names must be unique")
    for name, site in sites.items():
        if not isinstance(site, dict):
            raise ValueError("frozen robot site must be an object")
        position = site.get("pos")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in position
            )
        ):
            raise ValueError("frozen robot site positions must be finite XYZ")
    rod_names: set[str] = set()
    for rod in rods:
        if not isinstance(rod, dict):
            raise ValueError("frozen robot rod must be an object")
        name = rod.get("name")
        endpoints = (rod.get("site1"), rod.get("site2"))
        if not isinstance(name, str) or not name or name in rod_names:
            raise ValueError("frozen robot rod names must be unique")
        if (
            any(endpoint not in names for endpoint in endpoints)
            or endpoints[0] == endpoints[1]
        ):
            raise ValueError("frozen robot rod endpoints must name two sites")
        rod_names.add(name)


def prepare_frozen_initial_robot(
    *,
    mesh_path: Path,
    template_path: Path,
    template_manifest_path: Path,
    output_dir: Path,
) -> dict:
    """Verify Mesh/template lineage and atomically publish the base robot."""
    mesh = Path(mesh_path).resolve()
    template = Path(template_path).resolve()
    template_manifest = Path(template_manifest_path).resolve()
    for path, label in (
        (mesh, "source Mesh"),
        (template, "frozen robot template"),
        (template_manifest, "frozen template manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    manifest = _load_object(template_manifest)
    if (
        manifest.get("schema") != "qianji.frozen_mesh_vgt_template"
        or manifest.get("schema_version") != "1.0.0"
    ):
        raise ValueError("frozen template manifest header is invalid")
    source = manifest.get("source_mesh")
    template_record = manifest.get("template")
    generation = manifest.get("qianji_generation")
    if not all(
        isinstance(item, dict)
        for item in (source, template_record, generation)
    ):
        raise ValueError("frozen template lineage is incomplete")
    assert isinstance(source, dict)
    assert isinstance(template_record, dict)
    assert isinstance(generation, dict)
    expected_mesh_hash = _validate_sha256(
        source.get("sha256"),
        "source Mesh hash",
    )
    expected_template_hash = _validate_sha256(
        template_record.get("sha256"),
        "robot template hash",
    )
    if _sha256(mesh) != expected_mesh_hash:
        raise ValueError("source Mesh hash differs from frozen VGT lineage")
    if _sha256(template) != expected_template_hash:
        raise ValueError("robot template hash differs from frozen manifest")
    try:
        expected_sites = int(template_record["sites"])
        expected_rods = int(template_record["rods"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen template topology is missing") from error
    if expected_sites <= 0 or expected_rods <= 0:
        raise ValueError("frozen template topology must be positive")
    commit = generation.get("repository_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("QianJi generation commit is missing")
    if not isinstance(generation.get("entrypoint"), str) or not isinstance(
        generation.get("parameters"),
        dict,
    ):
        raise ValueError("QianJi generation command is incomplete")

    robot = _load_object(template)
    _validate_robot_topology(
        robot,
        expected_sites=expected_sites,
        expected_rods=expected_rods,
    )
    lineage = {
        "schema": "qianji.initial_vgt_model",
        "schema_version": "1.0.0",
        "source_mesh": {
            "path": str(mesh),
            "sha256": expected_mesh_hash,
        },
        "template": {
            "sha256": expected_template_hash,
        },
        "qianji_generation": generation,
        "artifact": {
            "path": "initial_model/robot_base.json",
            "sha256": expected_template_hash,
            "sites": expected_sites,
            "rods": expected_rods,
        },
        "frozen_precomputed_from_mesh": True,
        "reason": (
            "Freeze the QianJi Mesh conversion so repeated motion runs keep "
            "the same VGT topology and node identities."
        ),
    }

    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    final_robot = output / "robot_base.json"
    final_manifest = output / "base_robot_manifest.json"
    with TemporaryDirectory(
        prefix=".frozen-vgt-staging-",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary)
        staged_robot = staging / final_robot.name
        staged_manifest = staging / final_manifest.name
        shutil.copyfile(template, staged_robot)
        staged_manifest.write_text(
            json.dumps(
                lineage,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        publish_staged_files(
            [staged_robot, staged_manifest],
            [final_robot, final_manifest],
            overwrite=False,
            conflict_hint="choose an empty case directory",
        )
    return lineage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lineage = prepare_frozen_initial_robot(
        mesh_path=args.mesh,
        template_path=args.template,
        template_manifest_path=args.template_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(lineage, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
