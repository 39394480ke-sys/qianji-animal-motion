#!/usr/bin/env python3
"""Record the selected canonical robot used to generate the final MJCF files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_conversion(case_root: Path) -> Path:
    root = Path(case_root).resolve()
    records = {
        "source_robot": root / "initial_model/robot.json",
        "robot_xml": root / "initial_model/robot.xml",
        "robot_scene_xml": root / "initial_model/robot_scene.xml",
    }
    for label, path in records.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")
    output = root / "initial_model/robot_conversion_manifest.json"
    if output.exists():
        raise FileExistsError(f"conversion manifest already exists: {output}")
    payload = {
        "schema": "qianji.selected_robot_conversion",
        "schema_version": "1.0.0",
        "converter": "QianJi mujoco_builder/json2xml_v7_perrod.py",
        "artifacts": {
            label: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for label, path in records.items()
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    args = parser.parse_args()
    print(record_conversion(args.case_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
