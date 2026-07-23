"""Convert FBX assets to validated GLB files with Assimp."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from qianji_animal_motion.artifact_io import publish_staged_files


@dataclass(frozen=True)
class MeshMetadata:
    meshes: int
    nodes: int
    primitives: int
    vertices: int
    faces: int
    materials: int
    textures: int
    animations: int
    bounds_min: list[float]
    bounds_max: list[float]
    size_bytes: int


def build_assimp_command(
    input_path: Path,
    output_path: Path,
    *,
    assimp: str = "assimp",
) -> list[str]:
    """Build an Assimp command that explicitly requests binary glTF 2.0."""
    return [assimp, "export", str(input_path), str(output_path), "-fglb2"]


def _glb_document(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"invalid GLB: file is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    if magic != b"glTF":
        raise ValueError(f"invalid GLB magic: {path}")
    if version != 2:
        raise ValueError(f"GLB version must be 2, got {version}")
    if declared_length != len(data):
        raise ValueError(
            f"GLB length mismatch: header says {declared_length}, got {len(data)}"
        )

    chunk_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON":
        raise ValueError("GLB first chunk must be JSON")
    chunk_end = 20 + chunk_length
    if chunk_end > len(data):
        raise ValueError("GLB JSON chunk extends past end of file")
    try:
        return json.loads(data[20:chunk_end].rstrip(b" \x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid GLB JSON chunk: {exc}") from exc


def probe_glb(path: Path) -> MeshMetadata:
    """Read geometry counts and bounds from a binary glTF 2.0 file."""
    path = Path(path)
    document = _glb_document(path)
    accessors = document.get("accessors", [])
    meshes = document.get("meshes", [])
    primitive_count = 0
    vertex_count = 0
    face_count = 0
    minima: list[list[float]] = []
    maxima: list[list[float]] = []

    for mesh_index, mesh in enumerate(meshes):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            context = f"mesh {mesh_index} primitive {primitive_index}"
            primitive_count += 1
            position_index = primitive.get("attributes", {}).get("POSITION")
            if (
                not isinstance(position_index, int)
                or not 0 <= position_index < len(accessors)
            ):
                raise ValueError(
                    f"invalid GLB mesh: {context} has no valid POSITION accessor"
                )
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError(
                    f"invalid GLB mesh: {context} is not a triangle primitive"
                )
            positions = accessors[position_index]
            if positions.get("type") != "VEC3":
                raise ValueError(
                    f"invalid GLB mesh: {context} POSITION accessor must be VEC3"
                )
            positions_count = int(positions.get("count", 0))
            if positions_count < 3:
                raise ValueError(
                    f"invalid GLB mesh: {context} has fewer than three vertices"
                )
            vertex_count += positions_count
            minimum = positions.get("min", [])
            maximum = positions.get("max", [])
            if len(minimum) != 3 or len(maximum) != 3:
                raise ValueError(
                    f"invalid GLB mesh: {context} POSITION bounds are required"
                )
            minimum = [float(value) for value in minimum]
            maximum = [float(value) for value in maximum]
            if not all(math.isfinite(value) for value in minimum + maximum):
                raise ValueError(
                    f"invalid GLB mesh: {context} POSITION bounds must be finite"
                )
            if any(low > high for low, high in zip(minimum, maximum, strict=True)):
                raise ValueError(
                    f"invalid GLB mesh: {context} POSITION bounds are reversed"
                )
            minima.append(minimum)
            maxima.append(maximum)

            indices_index = primitive.get("indices")
            if indices_index is not None:
                if (
                    not isinstance(indices_index, int)
                    or not 0 <= indices_index < len(accessors)
                ):
                    raise ValueError(
                        f"invalid GLB mesh: {context} has an invalid index accessor"
                    )
                index_count = int(accessors[indices_index].get("count", 0))
                if index_count < 3 or index_count % 3:
                    raise ValueError(
                        f"invalid GLB mesh: {context} index count is not triangles"
                    )
                face_count += index_count // 3
            else:
                if positions_count % 3:
                    raise ValueError(
                        f"invalid GLB mesh: {context} vertex count is not triangles"
                    )
                face_count += positions_count // 3

    bounds_min = [min(values) for values in zip(*minima)] if minima else []
    bounds_max = [max(values) for values in zip(*maxima)] if maxima else []
    metadata = MeshMetadata(
        meshes=len(meshes),
        nodes=len(document.get("nodes", [])),
        primitives=primitive_count,
        vertices=vertex_count,
        faces=face_count,
        materials=len(document.get("materials", [])),
        textures=len(document.get("textures", [])),
        animations=len(document.get("animations", [])),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        size_bytes=path.stat().st_size,
    )
    errors = validate_mesh_metadata(metadata)
    if errors:
        raise ValueError("invalid GLB mesh: " + "; ".join(errors))
    return metadata


def validate_mesh_metadata(metadata: MeshMetadata) -> list[str]:
    """Return geometry problems that make a GLB unsuitable for QianJi."""
    errors: list[str] = []
    if metadata.meshes < 1:
        errors.append("at least one mesh is required")
    if metadata.nodes < 1:
        errors.append("at least one node is required")
    if metadata.primitives < 1:
        errors.append("at least one mesh primitive is required")
    if metadata.vertices < 3:
        errors.append("at least three vertices are required")
    if metadata.faces < 1:
        errors.append("at least one triangle face is required")
    bounds = metadata.bounds_min + metadata.bounds_max
    if len(metadata.bounds_min) != 3 or len(metadata.bounds_max) != 3:
        errors.append("POSITION bounds are required")
    elif not all(math.isfinite(value) for value in bounds):
        errors.append("POSITION bounds must be finite")
    return errors


def assimp_version(assimp: str = "assimp") -> str:
    """Return the installed Assimp version string."""
    try:
        completed = subprocess.run(
            [assimp, "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Assimp was not found; install it with `brew install assimp`"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
        raise RuntimeError(f"could not query Assimp version: {detail}") from exc
    match = re.search(r"\bVersion\s+([0-9]+(?:\.[0-9]+)*)", completed.stdout)
    return match.group(1) if match else "unknown"


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.{uuid.uuid4().hex}.tmp.glb"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_mesh(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
    assimp: str = "assimp",
) -> MeshMetadata:
    """Convert one FBX file to GLB, validate it, and write a metadata sidecar."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input FBX does not exist: {input_path}")
    if input_path.suffix.lower() != ".fbx":
        raise ValueError(f"input must be an FBX file: {input_path}")
    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"output must use the .glb extension: {output_path}")
    if input_path == output_path:
        raise ValueError("input and output paths must be different")
    sidecar = output_path.with_suffix(".metadata.json")
    existing = [path for path in (output_path, sidecar) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output already exists: {existing[0]}; use --overwrite to replace it"
        )

    input_hash = _sha256(input_path)
    version = assimp_version(assimp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_path.stem}.staging-",
        dir=output_path.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        staged_output = staging_dir / output_path.name
        staged_sidecar = staging_dir / sidecar.name
        temporary_path = _temporary_output_path(staged_output)
        command = build_assimp_command(input_path, temporary_path, assimp=assimp)
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            metadata = probe_glb(temporary_path)
            temporary_path.replace(staged_output)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Assimp was not found; install it with `brew install assimp`"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
            raise RuntimeError(f"Assimp conversion failed: {detail}") from exc
        finally:
            temporary_path.unlink(missing_ok=True)

        staged_sidecar.write_text(
            json.dumps(
                {
                    "schema": "qianji.mesh_convert",
                    "schema_version": "1.0.0",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "converter": {"name": "assimp", "version": version},
                    "input": {
                        "format": "fbx",
                        "size_bytes": input_path.stat().st_size,
                        "sha256": input_hash,
                    },
                    "output": asdict(metadata),
                },
                ensure_ascii=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if _sha256(input_path) != input_hash:
            raise RuntimeError("source FBX changed during conversion")
        publish_staged_files(
            (staged_output, staged_sidecar),
            (output_path, sidecar),
            overwrite=overwrite,
            conflict_hint="use --overwrite to replace it",
        )
    return metadata


def _default_output(source: Path) -> Path:
    if source.is_dir():
        return source.parent / "converted"
    return source.with_suffix(".glb")


def _fbx_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".fbx"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert FBX assets to validated binary glTF 2.0 files."
    )
    parser.add_argument("input", type=Path, help="Input FBX file or directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output GLB for one file, or output directory for batch mode",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--assimp", default="assimp", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.input.expanduser().resolve()
    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output(source)
    )

    if source.is_dir():
        files = _fbx_files(source)
        if not files:
            raise SystemExit(f"no FBX files found in {source}")
        destination.mkdir(parents=True, exist_ok=True)
        for fbx in files:
            output = destination / f"{fbx.stem}.glb"
            metadata = convert_mesh(
                fbx,
                output,
                overwrite=args.overwrite,
                assimp=args.assimp,
            )
            print(
                f"converted {fbx.name} -> {output.name} "
                f"({metadata.vertices} vertices, {metadata.faces} faces)"
            )
        return 0

    metadata = convert_mesh(
        source,
        destination,
        overwrite=args.overwrite,
        assimp=args.assimp,
    )
    print(
        f"converted {source.name} -> {destination} "
        f"({metadata.vertices} vertices, {metadata.faces} faces)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
