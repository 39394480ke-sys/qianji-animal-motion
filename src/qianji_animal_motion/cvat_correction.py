"""Safe CVAT round-trip support for manual six-keypoint correction."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

from qianji_animal_motion.semantic_cli import probe_video, render_preview
from qianji_animal_motion.semantic_mapping import MappingResult


KEYPOINT_NAMES = (
    "spine_front",
    "spine_rear",
    "front_left_foot",
    "front_right_foot",
    "rear_left_foot",
    "rear_right_foot",
)
SKELETON_LABEL = "quadruped_6"
_OWNED_EXPORT_FILES = (
    "annotations.xml",
    "cvat_manifest.json",
    "review_queue.json",
    "skeleton_definition.json",
)
_OWNED_IMPORT_FILES = (
    "keypoint_trajectory_2d_corrected.json",
    "corrections.json",
    "correction_report.json",
    "six_keypoints_corrected_preview.mp4",
)


@dataclass(frozen=True)
class CvatExportPaths:
    annotations: Path
    manifest: Path
    review_queue: Path
    skeleton_definition: Path


@dataclass(frozen=True)
class CvatImportPaths:
    trajectory: Path
    corrections: Path
    report: Path
    preview: Path | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_trajectory(trajectory: dict) -> None:
    if trajectory.get("schema") != "qianji.keypoint_trajectory_2d":
        raise ValueError("unsupported trajectory schema")
    video = trajectory.get("video", {})
    frames = trajectory.get("frames", [])
    if video.get("frame_count") != len(frames):
        raise ValueError("trajectory frame count does not match video metadata")
    for expected_index, frame in enumerate(frames):
        if frame.get("frame_idx") != expected_index:
            raise ValueError("trajectory frames must be contiguous and zero-based")
        names = tuple(frame.get("keypoints", {}).keys())
        if set(names) != set(KEYPOINT_NAMES):
            raise ValueError(f"frame {expected_index} does not contain the six keypoints")


def _ensure_output_available(output_dir: Path, owned_names: Sequence[str]) -> None:
    conflicts = [output_dir / name for name in owned_names if (output_dir / name).exists()]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"refusing to overwrite correction artifacts: {joined}; use a new output directory"
        )


def build_cvat_xml(trajectory: dict) -> str:
    """Build dense CVAT Video 1.1 skeleton annotations without interpolation."""
    _validate_trajectory(trajectory)
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "qianji-six-keypoint-correction"
    ET.SubElement(task, "size").text = str(len(trajectory["frames"]))
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(len(trajectory["frames"]) - 1)
    ET.SubElement(task, "frame_filter")
    ET.SubElement(task, "z_order").text = "False"

    track = ET.SubElement(
        root,
        "track",
        {"id": "0", "label": SKELETON_LABEL, "source": "auto"},
    )
    for frame in trajectory["frames"]:
        skeleton = ET.SubElement(
            track,
            "skeleton",
            {
                "frame": str(frame["frame_idx"]),
                "keyframe": "1",
                "z_order": "0",
            },
        )
        for name in KEYPOINT_NAMES:
            point = frame["keypoints"][name]
            valid = bool(point.get("valid"))
            x = float(point["x_px"]) if valid else 0.0
            y = float(point["y_px"]) if valid else 0.0
            ET.SubElement(
                skeleton,
                "points",
                {
                    "label": name,
                    "outside": "0" if valid else "1",
                    "occluded": "0",
                    "keyframe": "1",
                    "points": f"{x:.6f},{y:.6f}",
                },
            )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def _contiguous_ranges(values: Sequence[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ordered = sorted(set(int(value) for value in values))
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ranges


def build_review_queue(
    trajectory: dict,
    report: dict,
    *,
    context_frames: int = 2,
) -> dict:
    """Create a compact, reasoned queue instead of asking for blind frame review."""
    _validate_trajectory(trajectory)
    frame_count = len(trajectory["frames"])
    reasons: dict[int, set[str]] = {}
    priorities: dict[int, str] = {}

    def add(frame_idx: int, reason: str, priority: str) -> None:
        if not 0 <= frame_idx < frame_count:
            return
        reasons.setdefault(frame_idx, set()).add(reason)
        if priority == "critical" or frame_idx not in priorities:
            priorities[frame_idx] = priority

    for name, point_report in report.get("keypoints", {}).items():
        for frame_idx in point_report.get("invalid_frames", []):
            point_flags = trajectory["frames"][frame_idx]["keypoints"][name].get(
                "flags", []
            )
            if point_flags:
                for flag in point_flags:
                    add(frame_idx, f"{name}:{flag}", "critical")
            else:
                add(frame_idx, f"{name}:invalid", "critical")

    for group, values in report.get("identity_ambiguous_frames", {}).items():
        for frame_idx in values:
            add(frame_idx, f"{group}_identity_ambiguous", "critical")

    for group, values in report.get("identity_corrections", {}).items():
        for start, end in _contiguous_ranges(values):
            add(start, f"{group}_identity_correction_boundary", "high")
            add(end, f"{group}_identity_correction_boundary", "high")

    direct_frames = sorted(reasons)
    for frame_idx in direct_frames:
        for offset in range(-context_frames, context_frames + 1):
            neighbor = frame_idx + offset
            if neighbor not in reasons and 0 <= neighbor < frame_count:
                reasons[neighbor] = {f"context_for_frame_{frame_idx}"}
                priorities[neighbor] = "context"

    ordered_priority = {"critical": 0, "high": 1, "context": 2}
    frames = [
        {
            "frame_idx": frame_idx,
            "timestamp_s": trajectory["frames"][frame_idx]["timestamp_s"],
            "priority": priorities[frame_idx],
            "context_only": priorities[frame_idx] == "context",
            "reasons": sorted(reasons[frame_idx]),
        }
        for frame_idx in sorted(
            reasons,
            key=lambda item: (ordered_priority[priorities[item]], item),
        )
    ]
    return {
        "schema": "qianji.keypoint_review_queue",
        "schema_version": "1.0.0",
        "context_frames": context_frames,
        "frames": frames,
    }


def _skeleton_definition() -> dict:
    return {
        "label": SKELETON_LABEL,
        "keypoints": list(KEYPOINT_NAMES),
        "edges": [
            ["spine_rear", "spine_front"],
            ["spine_front", "front_left_foot"],
            ["spine_front", "front_right_foot"],
            ["spine_rear", "rear_left_foot"],
            ["spine_rear", "rear_right_foot"],
        ],
        "note": "Create this skeleton label in CVAT before importing annotations.xml.",
    }


def export_cvat_package(
    *,
    video_path: Path,
    trajectory_path: Path,
    report_path: Path,
    output_dir: Path,
    skip_video_probe: bool = False,
) -> CvatExportPaths:
    video_path = Path(video_path).resolve()
    trajectory_path = Path(trajectory_path).resolve()
    report_path = Path(report_path).resolve()
    output_dir = Path(output_dir)
    trajectory = _load_json(trajectory_path)
    report = _load_json(report_path)
    _validate_trajectory(trajectory)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not skip_video_probe:
        actual = probe_video(video_path)
        expected = trajectory["video"]
        if (
            actual.width != expected["width"]
            or actual.height != expected["height"]
            or actual.frame_count != expected["frame_count"]
            or abs(actual.fps - expected["fps"]) > 0.01
        ):
            raise ValueError("video metadata does not match trajectory")

    _ensure_output_available(output_dir, _OWNED_EXPORT_FILES)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = CvatExportPaths(
        annotations=output_dir / "annotations.xml",
        manifest=output_dir / "cvat_manifest.json",
        review_queue=output_dir / "review_queue.json",
        skeleton_definition=output_dir / "skeleton_definition.json",
    )
    manifest = {
        "schema": "qianji.cvat_correction_manifest",
        "schema_version": "1.0.0",
        "created_at": _utc_now(),
        "video_path": str(video_path),
        "video_sha256": _sha256(video_path),
        "baseline_trajectory_path": str(trajectory_path),
        "baseline_trajectory_sha256": _sha256(trajectory_path),
        "mapping_report_path": str(report_path),
        "mapping_report_sha256": _sha256(report_path),
        "predictions_sha256": trajectory.get("source", {}).get(
            "predictions_sha256"
        ),
        "video": trajectory["video"],
        "keypoint_names": list(KEYPOINT_NAMES),
    }
    paths.annotations.write_text(build_cvat_xml(trajectory), encoding="utf-8")
    _write_json(paths.manifest, manifest)
    _write_json(paths.review_queue, build_review_queue(trajectory, report))
    _write_json(paths.skeleton_definition, _skeleton_definition())
    return paths


def _parse_point_coordinates(value: str) -> tuple[float, float]:
    pieces = value.split(";")
    if len(pieces) != 1:
        raise ValueError("each skeleton element must contain exactly one point")
    coordinates = pieces[0].split(",")
    if len(coordinates) != 2:
        raise ValueError("invalid CVAT point coordinates")
    return float(coordinates[0]), float(coordinates[1])


def _parse_cvat_frames(xml_text: str, frame_count: int) -> dict[int, dict[str, dict]]:
    root = ET.fromstring(xml_text)
    tracks = root.findall(f"./track[@label='{SKELETON_LABEL}']")
    if len(tracks) != 1:
        raise ValueError(f"expected one {SKELETON_LABEL} track")
    result: dict[int, dict[str, dict]] = {}
    for skeleton in tracks[0].findall("./skeleton"):
        frame_idx = int(skeleton.attrib["frame"])
        if frame_idx in result:
            raise ValueError(f"duplicate CVAT skeleton at frame {frame_idx}")
        points: dict[str, dict] = {}
        for element in skeleton.findall("./points"):
            name = element.attrib["label"]
            if name in points:
                raise ValueError(f"duplicate {name} at frame {frame_idx}")
            points[name] = {
                "xy": _parse_point_coordinates(element.attrib["points"]),
                "outside": element.attrib.get("outside", "0") == "1",
                "occluded": element.attrib.get("occluded", "0") == "1",
            }
        if set(points) != set(KEYPOINT_NAMES):
            raise ValueError(f"frame {frame_idx} does not contain the six CVAT points")
        result[frame_idx] = points
    if set(result) != set(range(frame_count)):
        raise ValueError("CVAT annotations must contain every frame explicitly")
    return result


def apply_cvat_corrections(
    baseline: dict,
    annotations_xml: str,
    *,
    coordinate_tolerance: float = 0.01,
) -> tuple[dict, dict, dict]:
    _validate_trajectory(baseline)
    parsed = _parse_cvat_frames(annotations_xml, len(baseline["frames"]))
    corrected = copy.deepcopy(baseline)
    changes: list[dict] = []

    for frame in corrected["frames"]:
        frame_idx = frame["frame_idx"]
        for name in KEYPOINT_NAMES:
            original = baseline["frames"][frame_idx]["keypoints"][name]
            output = frame["keypoints"][name]
            cvat = parsed[frame_idx][name]
            original_valid = bool(original["valid"])
            if cvat["outside"]:
                if not original_valid:
                    continue
                before = {
                    "x_px": original["x_px"],
                    "y_px": original["y_px"],
                    "valid": True,
                }
                output["x_px"] = None
                output["y_px"] = None
                output["valid"] = False
                output["position_source"] = "manual"
                output["review_status"] = "unresolvable"
                output["flags"] = list(
                    dict.fromkeys([*original.get("flags", []), "manual_marked_missing"])
                )
                output["correction"] = {"original": before}
                changes.append(
                    {
                        "frame_idx": frame_idx,
                        "keypoint": name,
                        "action": "mark_missing",
                        "before": before,
                        "after": {"x_px": None, "y_px": None, "valid": False},
                    }
                )
                continue

            x, y = cvat["xy"]
            width = float(baseline["video"]["width"])
            height = float(baseline["video"]["height"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"non-finite coordinates for {name} at frame {frame_idx}")
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError(
                    f"coordinates outside video bounds for {name} at frame {frame_idx}"
                )
            if not original_valid and x == 0.0 and y == 0.0:
                raise ValueError(
                    f"placeholder coordinates were made visible for {name} at frame "
                    f"{frame_idx}; move the point before clearing Outside"
                )
            moved = (
                not original_valid
                or abs(float(original["x_px"]) - x) > coordinate_tolerance
                or abs(float(original["y_px"]) - y) > coordinate_tolerance
            )
            occlusion_changed = cvat["occluded"]
            if not moved and not occlusion_changed:
                continue

            before = {
                "x_px": original["x_px"],
                "y_px": original["y_px"],
                "valid": original_valid,
            }
            action = "recover" if not original_valid else "move"
            output["x_px"] = x
            output["y_px"] = y
            output["valid"] = True
            output["position_source"] = "manual"
            output["review_status"] = "occluded" if cvat["occluded"] else "corrected"
            output["flags"] = ["manually_corrected"]
            if cvat["occluded"]:
                output["flags"].append("manual_occluded")
            output["correction"] = {
                "original": before,
                "original_flags": list(original.get("flags", [])),
            }
            changes.append(
                {
                    "frame_idx": frame_idx,
                    "keypoint": name,
                    "action": action,
                    "before": before,
                    "after": {"x_px": x, "y_px": y, "valid": True},
                    "occluded": cvat["occluded"],
                }
            )

    if changes:
        corrected["schema_version"] = "1.2.0"
        corrected["manual_correction"] = {
            "applied": True,
            "policy": "manual_only_no_interpolation",
            "changed_points": len(changes),
        }
    corrections = {
        "schema": "qianji.keypoint_corrections",
        "schema_version": "1.0.0",
        "policy": "manual_only_no_interpolation",
        "corrections": changes,
    }
    remaining_invalid = {
        name: [
            frame["frame_idx"]
            for frame in corrected["frames"]
            if not frame["keypoints"][name]["valid"]
        ]
        for name in KEYPOINT_NAMES
    }
    report = {
        "schema": "qianji.keypoint_correction_report",
        "schema_version": "1.0.0",
        "changed_points": len(changes),
        "moved_points": sum(change["action"] == "move" for change in changes),
        "recovered_points": sum(change["action"] == "recover" for change in changes),
        "manually_missing_points": sum(
            change["action"] == "mark_missing" for change in changes
        ),
        "remaining_invalid_frames": remaining_invalid,
        "interpolation_used": False,
    }
    return corrected, corrections, report


def import_cvat_package(
    *,
    baseline_path: Path,
    manifest: dict | Path,
    annotations_xml: str | Path,
    output_dir: Path,
    video_path: Path | None = None,
    render_video: bool = True,
) -> CvatImportPaths:
    baseline_path = Path(baseline_path).resolve()
    output_dir = Path(output_dir)
    manifest_payload = _load_json(manifest) if isinstance(manifest, Path) else manifest
    if manifest_payload.get("baseline_trajectory_sha256") != _sha256(baseline_path):
        raise ValueError("baseline trajectory hash does not match CVAT manifest")
    baseline = _load_json(baseline_path)
    if isinstance(annotations_xml, Path):
        xml_text = annotations_xml.read_text(encoding="utf-8")
    else:
        xml_text = annotations_xml
    corrected, corrections, report = apply_cvat_corrections(baseline, xml_text)

    if video_path is not None:
        video_path = Path(video_path).resolve()
        expected_video_hash = manifest_payload.get("video_sha256")
        if expected_video_hash and _sha256(video_path) != expected_video_hash:
            raise ValueError("video hash does not match CVAT manifest")
    if render_video and video_path is None:
        raise ValueError("video path is required when rendering a preview")

    owned = list(_OWNED_IMPORT_FILES)
    if not render_video:
        owned.remove("six_keypoints_corrected_preview.mp4")
    _ensure_output_available(output_dir, owned)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = CvatImportPaths(
        trajectory=output_dir / "keypoint_trajectory_2d_corrected.json",
        corrections=output_dir / "corrections.json",
        report=output_dir / "correction_report.json",
        preview=(
            output_dir / "six_keypoints_corrected_preview.mp4"
            if render_video
            else None
        ),
    )
    _write_json(paths.trajectory, corrected)
    _write_json(paths.corrections, corrections)
    report["baseline_trajectory_sha256"] = _sha256(baseline_path)
    report["manifest_schema_version"] = manifest_payload.get("schema_version")
    _write_json(paths.report, report)
    if render_video and paths.preview is not None and video_path is not None:
        render_preview(
            video_path,
            MappingResult(trajectory=corrected, report=report),
            paths.preview,
        )
    return paths


def _export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a six-point trajectory to CVAT")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def export_main(argv: Sequence[str] | None = None) -> int:
    args = _export_parser().parse_args(argv)
    paths = export_cvat_package(
        video_path=args.video,
        trajectory_path=args.trajectory,
        report_path=args.report,
        output_dir=args.output,
    )
    for path in paths.__dict__.values():
        print(path)
    return 0


def _import_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import manually corrected CVAT points")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def import_main(argv: Sequence[str] | None = None) -> int:
    args = _import_parser().parse_args(argv)
    paths = import_cvat_package(
        baseline_path=args.baseline,
        manifest=args.manifest,
        annotations_xml=args.annotations,
        video_path=args.video,
        output_dir=args.output,
    )
    for path in paths.__dict__.values():
        if path is not None:
            print(path)
    return 0
