import json
import stat
import struct
from pathlib import Path
from typing import Callable

import pytest

from qianji_animal_motion.mesh_convert import (
    build_assimp_command,
    convert_mesh,
    probe_glb,
)


def _write_triangle_glb(
    path: Path,
    mutate: Callable[[dict], None] | None = None,
) -> None:
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                ]
            }
        ],
        "buffers": [{"byteLength": 44}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ],
    }
    if mutate is not None:
        mutate(document)
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    binary_chunk = positions + struct.pack("<3H", 0, 1, 2) + b"\x00\x00"
    length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(binary_chunk), b"BIN\x00")
        + binary_chunk
    )


def _write_fake_assimp(path: Path, fixture: Path, *, fail: bool = False) -> None:
    body = f"""#!/usr/bin/env python3
import shutil
import sys

if sys.argv[1] == "version":
    print("Version 6.0 -shared -st")
    raise SystemExit(0)
if sys.argv[1] == "export":
    shutil.copyfile({str(fixture)!r}, sys.argv[3])
    raise SystemExit({7 if fail else 0})
raise SystemExit(2)
"""
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_build_assimp_command_requests_binary_gltf(tmp_path: Path) -> None:
    source = tmp_path / "cat.fbx"
    output = tmp_path / "cat.glb"

    command = build_assimp_command(source, output)

    assert command == ["assimp", "export", str(source), str(output), "-fglb2"]


def test_probe_glb_reports_triangle_mesh(tmp_path: Path) -> None:
    glb = tmp_path / "triangle.glb"
    _write_triangle_glb(glb)

    metadata = probe_glb(glb)

    assert metadata.meshes == 1
    assert metadata.nodes == 1
    assert metadata.primitives == 1
    assert metadata.vertices == 3
    assert metadata.faces == 1
    assert metadata.bounds_min == [0.0, 0.0, 0.0]
    assert metadata.bounds_max == [1.0, 1.0, 0.0]


def test_probe_glb_rejects_primitive_without_position_accessor(
    tmp_path: Path,
) -> None:
    glb = tmp_path / "missing-position.glb"

    def remove_position(document: dict) -> None:
        document["meshes"][0]["primitives"][0]["attributes"] = {}

    _write_triangle_glb(glb, remove_position)

    with pytest.raises(ValueError, match="POSITION accessor"):
        probe_glb(glb)


def test_probe_glb_rejects_reversed_position_bounds(tmp_path: Path) -> None:
    glb = tmp_path / "reversed-bounds.glb"

    def reverse_bounds(document: dict) -> None:
        document["accessors"][0]["min"] = [2.0, 0.0, 0.0]
        document["accessors"][0]["max"] = [1.0, 1.0, 0.0]

    _write_triangle_glb(glb, reverse_bounds)

    with pytest.raises(ValueError, match="POSITION bounds"):
        probe_glb(glb)


def test_convert_mesh_handles_unicode_path_and_writes_metadata(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.glb"
    _write_triangle_glb(fixture)
    assimp = tmp_path / "fake_assimp"
    _write_fake_assimp(assimp, fixture)
    source = tmp_path / "小猫.fbx"
    source.write_bytes(b"Kaydara FBX Binary")
    output = tmp_path / "小猫.glb"

    result = convert_mesh(source, output, assimp=str(assimp))

    assert result.meshes == 1
    assert output.is_file()
    payload = json.loads(
        output.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "qianji.mesh_convert"
    assert payload["converter"]["name"] == "assimp"
    assert payload["converter"]["version"] == "6.0"
    assert payload["output"]["vertices"] == 3


def test_convert_mesh_keeps_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "cat.fbx"
    source.write_bytes(b"fbx")
    output = tmp_path / "cat.glb"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_mesh(source, output)

    assert output.read_bytes() == b"existing"


def test_convert_mesh_removes_temporary_output_after_failure(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.glb"
    _write_triangle_glb(fixture)
    assimp = tmp_path / "failing_assimp"
    _write_fake_assimp(assimp, fixture, fail=True)
    source = tmp_path / "cat.fbx"
    source.write_bytes(b"fbx")
    output = tmp_path / "cat.glb"

    with pytest.raises(RuntimeError, match="Assimp conversion failed"):
        convert_mesh(source, output, assimp=str(assimp))

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp.glb"))


def test_convert_mesh_preserves_existing_pair_when_sidecar_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "fixture.glb"
    _write_triangle_glb(fixture)
    assimp = tmp_path / "fake_assimp"
    _write_fake_assimp(assimp, fixture)
    source = tmp_path / "cat.fbx"
    source.write_bytes(b"fbx")
    output = tmp_path / "cat.glb"
    sidecar = output.with_suffix(".metadata.json")
    output.write_bytes(b"old glb")
    sidecar.write_bytes(b"old metadata")
    original_write_text = Path.write_text

    def fail_sidecar(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == sidecar.name:
            raise OSError("sidecar write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_sidecar)

    with pytest.raises(OSError, match="sidecar write failed"):
        convert_mesh(
            source,
            output,
            overwrite=True,
            assimp=str(assimp),
        )

    assert output.read_bytes() == b"old glb"
    assert sidecar.read_bytes() == b"old metadata"
