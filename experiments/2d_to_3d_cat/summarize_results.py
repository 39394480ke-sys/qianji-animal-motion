"""Summarize the reproducible cat mesh-retargeting scale sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCALES = (0.10, 0.25, 0.50)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_case_path(path: Path, output_root: Path) -> str:
    return path.resolve().relative_to(output_root).as_posix()


def _validated_source_hashes(
    sources: object,
    output_root: Path,
) -> dict[str, str]:
    expected_labels = ("trajectory", "robot", "rig")
    if not isinstance(sources, dict) or set(sources) != set(expected_labels):
        raise ValueError("scale lift sources must contain trajectory, robot, and rig")
    hashes = {}
    for label in expected_labels:
        item = sources[label]
        if not isinstance(item, dict):
            raise ValueError(f"scale lift source {label} is malformed")
        raw_path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"scale lift source {label} is malformed")
        path = Path(raw_path)
        if not path.is_absolute():
            path = output_root / path
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"scale lift source {label} hash differs")
        hashes[label] = digest
    return hashes


def summarize(output_root: Path, mesh: Path, trajectory: Path) -> dict:
    output_root = output_root.resolve()
    robot_path = output_root / "morphology/robot.json"
    rig_path = output_root / "rig_seed/rig_keypoints.json"
    robot = _load(robot_path)
    rig = _load(rig_path)
    rigidity = _load(
        output_root / "morphology/reports/rigidity_report_3d.json"
    )

    scale_results = []
    verified_scale_source_hashes = {
        "trajectory": _sha256(trajectory),
        "robot": _sha256(robot_path),
        "rig": _sha256(rig_path),
    }
    verified_lineage_hashes: dict[str, str] | None = None
    for scale in SCALES:
        scale_name = f"{scale:.2f}"
        scale_root = output_root / f"scale_{scale_name}"
        lift = _load(scale_root / "lift_report.json")
        if (
            _validated_source_hashes(lift.get("sources"), output_root)
            != verified_scale_source_hashes
        ):
            raise ValueError("scale lift sources differ across the sweep")
        lineage = lift.get("verified_lineage")
        if lineage:
            if not isinstance(lineage, dict):
                raise ValueError("scale verified lineage is malformed")
            current_lineage = {
                label: item.get("sha256")
                for label, item in lineage.items()
                if isinstance(item, dict)
            }
            if len(current_lineage) != len(lineage) or any(
                not isinstance(digest, str) or len(digest) != 64
                for digest in current_lineage.values()
            ):
                raise ValueError("scale verified lineage is malformed")
            if verified_lineage_hashes is None:
                verified_lineage_hashes = current_lineage
            elif current_lineage != verified_lineage_hashes:
                raise ValueError("scale verified lineage differs across the sweep")
        run = _load(scale_root / "reachability/run_summary.json")
        preview = _load(
            scale_root / "preview/keypoint_motion_preview_report.json"
        )
        summary = run["summary"]
        scale_results.append(
            {
                "motion_scale": scale,
                "lift": {
                    "reference_frame": lift["reference_frame"],
                    "substitution_counts": lift["substitution_counts"],
                    "displacement_norm": lift["displacement_norm"],
                },
                "reachability": summary,
                "preview": {
                    "frames": preview["n_frames"],
                    "duration_s": preview["duration_s"],
                    "overview_png": preview["outputs"]["overview_png"],
                    "three_view_png": preview["outputs"]["three_view_png"],
                    "preview_mp4": preview["outputs"]["preview_mp4"],
                },
            }
        )

    fully_non_unreachable = [
        item
        for item in scale_results
        if item["reachability"]["marginal_or_feasible_fraction"] == 1.0
    ]
    if fully_non_unreachable:
        selected = max(
            fully_non_unreachable,
            key=lambda item: (
                item["motion_scale"],
                -item["reachability"]["max_keypoint_error_m"],
            ),
        )
        selection_reason = (
            "highest tested scale with no unreachable frames"
        )
    else:
        selected = min(
            scale_results,
            key=lambda item: item["reachability"]["max_keypoint_error_m"],
        )
        selection_reason = (
            "no tested scale avoided unreachable frames; selected lowest error"
        )

    return {
        "schema": "qianji.cat_2d_to_3d_experiment_summary",
        "schema_version": "0.1.0",
        "sources": {
            "mesh": {
                "path": str(mesh.resolve()),
                "sha256": _sha256(mesh),
            },
            "trajectory": {
                "path": str(trajectory.resolve()),
                "sha256": _sha256(trajectory),
            },
            "robot": {
                "path": _relative_case_path(robot_path, output_root),
                "sha256": _sha256(robot_path),
            },
            "rig": {
                "path": _relative_case_path(rig_path, output_root),
                "sha256": _sha256(rig_path),
            },
        },
        "verified_scale_source_hashes": verified_scale_source_hashes,
        "verified_lineage_hashes": verified_lineage_hashes or {},
        "morphology": {
            "name": robot.get("name"),
            "sites": len(robot["sites"]),
            "rods": len(robot["rod_groups"]),
            "rank": rigidity["rank"],
            "target_rank": rigidity["target_rank"],
            "max_node_degree": rigidity["node_degree_report"]["max_degree"],
            "rig_roles": len(rig["key_site_map"]),
        },
        "scale_results": scale_results,
        "selected_motion_scale": selected["motion_scale"],
        "selection_reason": selection_reason,
        "limitations": [
            "body-relative 2.5D retarget, not metric monocular 3D",
            "camera depth and global translation are unobserved",
            "reachability is geometric and does not prove dynamic stability",
        ],
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# Cat 2D-to-3D Mesh Experiment",
        "",
        (
            f"Selected scale: `{summary['selected_motion_scale']:.2f}` "
            f"({summary['selection_reason']})."
        ),
        "",
        "| Scale | Feasible | Marginal | Unreachable | Max error (m) | Mean error (m) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["scale_results"]:
        reachability = item["reachability"]
        counts = reachability["status_counts"]
        lines.append(
            f"| {item['motion_scale']:.2f} "
            f"| {counts.get('feasible', 0)} "
            f"| {counts.get('marginal', 0)} "
            f"| {counts.get('unreachable', 0)} "
            f"| {reachability['max_keypoint_error_m']:.6f} "
            f"| {reachability['mean_keypoint_error_m']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is body-relative 2.5D retargeting. Camera depth, camera",
            "calibration, global translation, and dynamic stability are not",
            "established by this experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.output_root, args.mesh, args.trajectory)
    json_path = args.output_root / "experiment_summary.json"
    markdown_path = args.output_root / "experiment_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
