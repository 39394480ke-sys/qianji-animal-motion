#!/usr/bin/env python3
"""Aggregate and select versioned QianJi reachability candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from qianji_animal_motion.vgt_sequence import load_and_validate_vgt_sequence
from qianji_animal_motion.vgt_sequence import (
    compute_rod_constraint_metrics,
    recompute_reachability_metrics,
    validate_control_pair_against_sequence,
    validate_reachability_report,
)
from qianji_animal_motion.vgt_control import (
    infer_uniform_contraction_fraction,
)


MAX_KEYPOINT_ERROR_M = 0.05
MAX_EDGE_VIOLATION_M = 0.0005
MAX_CLIPPED_FRACTION = 0.05


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate artifact {label!r} must name a path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"candidate artifact {label!r} is missing: {path}")
    return path


def _finite_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _candidate_record(candidate_root: Path) -> dict:
    root = Path(candidate_root).resolve()
    metadata_path = root / "candidate.json"
    metadata = _load_json(metadata_path)
    artifacts = metadata.get("artifacts")
    parameters = metadata.get("parameters")
    if not isinstance(artifacts, dict) or not isinstance(parameters, dict):
        raise ValueError("candidate.json must contain artifacts and parameters")
    paths = {
        label: _resolve(root, artifacts.get(label), label)
        for label in (
            "run_summary",
            "reachability_report",
            "robot",
            "rig",
            "desired_motion",
            "projected_motion",
            "site_npz",
        )
    }
    if paths["desired_motion"] == paths["projected_motion"]:
        raise ValueError("desired and projected candidate motions must differ")
    run_summary = _load_json(paths["run_summary"])
    reachability = _load_json(paths["reachability_report"])
    robot = _load_json(paths["robot"])
    rig = _load_json(paths["rig"])
    desired = _load_json(paths["desired_motion"])
    projected = _load_json(paths["projected_motion"])
    summary = reachability.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("reachability report is missing summary")
    if run_summary.get("summary") != summary:
        raise ValueError("run summary and reachability summary differ")
    try:
        frames = int(summary["frames"])
        status_counts = summary["status_counts"]
        unreachable = int(status_counts["unreachable"])
        feasible_fraction = _finite_number(
            summary["feasible_fraction"],
            "feasible fraction",
        )
        max_error = _finite_number(
            summary["max_keypoint_error_m"],
            "maximum keypoint error",
        )
        max_edge_violation = _finite_number(
            summary["max_edge_violation_m"],
            "maximum edge violation",
        )
        max_clipped_fraction = _finite_number(
            summary["max_estimated_clipped_fraction"],
            "maximum estimated clipped fraction",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate reachability summary is malformed") from error
    if frames <= 0 or not isinstance(status_counts, dict):
        raise ValueError("candidate reachability frame/status counts are invalid")
    if sum(int(status_counts.get(name, 0)) for name in (
        "feasible",
        "marginal",
        "unreachable",
    )) != frames:
        raise ValueError("candidate status counts do not sum to frame count")
    sequence = load_and_validate_vgt_sequence(
        paths["site_npz"],
        robot,
        expected_frames=frames,
        expected_sites=12,
        expected_rods=30,
    )
    if rig.get("schema") != "qianji-key-site-map":
        raise ValueError("candidate rig schema is unsupported")
    if desired.get("schema") != "qianji-keypoint-trajectory-v1":
        raise ValueError("candidate desired motion schema is unsupported")
    if projected.get("schema") != "qianji-keypoint-trajectory-v1":
        raise ValueError("candidate projected motion schema is unsupported")
    if len(desired.get("frames", ())) != frames or len(
        projected.get("frames", ())
    ) != frames:
        raise ValueError("candidate control motion frame count differs")
    try:
        fps = float(desired["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate desired motion fps is malformed") from error
    desired_motion, projected_motion = validate_control_pair_against_sequence(
        desired,
        projected,
        sequence,
        rig,
        expected_frames=frames,
        expected_fps=fps,
    )
    recomputed = recompute_reachability_metrics(
        desired_motion,
        projected_motion,
        sequence,
        robot,
    )
    validate_reachability_report(
        reachability,
        recomputed,
        desired_motion,
        projected_motion,
    )
    summary = recomputed["summary"]

    motion_scale = _finite_number(parameters.get("motion_scale"), "motion scale")
    declared_contraction = _finite_number(
        parameters.get("contraction_fraction"),
        "contraction fraction",
    )
    actual_contraction = infer_uniform_contraction_fraction(robot)
    if not math.isclose(
        declared_contraction,
        actual_contraction,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "candidate declared contraction does not match robot constraints"
        )
    extension_only = reachability.get("extension_only")
    expected_extension_only = actual_contraction <= 1e-12
    if extension_only is not expected_extension_only:
        raise ValueError(
            "reachability extension_only disagrees with robot contraction"
        )
    geometry = recomputed["geometry"]
    rig_variant = parameters.get("rig_variant")
    morphology = parameters.get("morphology_variant")
    if not isinstance(rig_variant, str) or not isinstance(morphology, str):
        raise ValueError("candidate rig and morphology variants must be strings")
    failures = []
    if frames != int(metadata.get("expected_frames", frames)):
        failures.append("frame_count")
    if unreachable != 0:
        failures.append("unreachable_frames")
    if max_error > MAX_KEYPOINT_ERROR_M:
        failures.append("max_keypoint_error_above_0.05_m")
    if max_edge_violation > MAX_EDGE_VIOLATION_M:
        failures.append("max_edge_violation_above_0.0005_m")
    if max_clipped_fraction > MAX_CLIPPED_FRACTION:
        failures.append(
            "max_estimated_clipped_fraction_above_0.05"
        )
    if geometry["max_edge_violation_m"] > MAX_EDGE_VIOLATION_M:
        failures.append("recomputed_edge_violation_above_0.0005_m")
    if (
        geometry["max_violated_rod_fraction"]
        > MAX_CLIPPED_FRACTION
    ):
        failures.append("recomputed_violated_rod_fraction_above_0.05")
    if sequence.positions.shape != (frames, 12, 3):
        failures.append("vgt_shape")
    hashes = {label: _sha256(path) for label, path in paths.items()}
    if hashes["desired_motion"] == hashes["projected_motion"]:
        failures.append("desired_projected_content_alias")
    record = {
        "candidate_id": metadata.get("candidate_id", root.name),
        "candidate_root": str(root),
        "motion_scale": motion_scale,
        "contraction_fraction": declared_contraction,
        "actual_contraction_fraction": actual_contraction,
        "extension_only": extension_only,
        "rig_variant": rig_variant,
        "morphology_variant": morphology,
        "summary": summary,
        "recomputed_reachability": {
            "thresholds": recomputed["thresholds"],
            "summary": summary,
        },
        "recomputed_geometry": geometry,
        "eligible": not failures,
        "eligibility_failures": failures,
        "artifacts": {
            label: {"path": str(path), "sha256": hashes[label]}
            for label, path in paths.items()
        },
    }
    morphology_report = artifacts.get("morphology_report")
    if morphology_report is not None:
        morphology_path = _resolve(root, morphology_report, "morphology_report")
        record["morphology_report"] = {
            "path": str(morphology_path),
            "sha256": _sha256(morphology_path),
            "payload": _load_json(morphology_path),
        }
    return record


def select_candidate(report: dict) -> dict:
    """Select highest scale, then feasible fraction, then lowest error."""
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("comparison report must contain candidates")
    eligible = [item for item in candidates if item.get("eligible") is True]
    if not eligible:
        raise ValueError("no reachability candidate meets the acceptance gates")

    def priority(item: dict) -> tuple[float, float, float, str]:
        summary = item["summary"]
        return (
            -float(item["motion_scale"]),
            -float(summary["feasible_fraction"]),
            float(summary["max_keypoint_error_m"]),
            str(item["candidate_id"]),
        )

    return min(eligible, key=priority)


def _group_comparison(candidates: list[dict], field: str) -> dict:
    values = {}
    for item in candidates:
        key = str(item[field])
        values.setdefault(key, []).append(
            {
                "candidate_id": item["candidate_id"],
                "eligible": item["eligible"],
                "feasible_fraction": item["summary"]["feasible_fraction"],
                "max_keypoint_error_m": item["summary"][
                    "max_keypoint_error_m"
                ],
                "unreachable": item["summary"]["status_counts"]["unreachable"],
            }
        )
    return values


def _case_relative(path: Path, case_root: Path) -> str:
    try:
        return path.resolve().relative_to(case_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"candidate artifact is outside the case root: {path}"
        ) from error


def _relativize_record(record: dict, case_root: Path) -> None:
    record["candidate_root"] = _case_relative(
        Path(record["candidate_root"]),
        case_root,
    )
    for item in record["artifacts"].values():
        item["path"] = _case_relative(Path(item["path"]), case_root)
    morphology = record.get("morphology_report")
    if isinstance(morphology, dict):
        morphology["path"] = _case_relative(
            Path(morphology["path"]),
            case_root,
        )


def compare_candidates(candidate_roots: list[Path]) -> dict:
    """Validate all artifacts, aggregate parameters, and select one candidate."""
    if not candidate_roots:
        raise ValueError("at least one candidate root is required")
    roots = [Path(root).resolve() for root in candidate_roots]
    candidate_parents = {root.parent for root in roots}
    if (
        len(candidate_parents) != 1
        or next(iter(candidate_parents)).name != "candidates"
    ):
        raise ValueError(
            "candidate roots must share one case-local candidates directory"
        )
    case_root = next(iter(candidate_parents)).parent
    candidates = [_candidate_record(root) for root in roots]
    for record in candidates:
        _relativize_record(record, case_root)
    ids = [item["candidate_id"] for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    report = {
        "schema": "qianji.39point_reachability_comparison",
        "schema_version": "0.1.0",
        "acceptance": {
            "unreachable_frames": 0,
            "max_keypoint_error_m": MAX_KEYPOINT_ERROR_M,
            "max_edge_violation_m": MAX_EDGE_VIOLATION_M,
            "max_estimated_clipped_fraction": MAX_CLIPPED_FRACTION,
            "max_recomputed_violated_rod_fraction": MAX_CLIPPED_FRACTION,
            "sites": 12,
            "rods": 30,
        },
        "candidates": candidates,
        "comparisons": {
            "motion_scale": _group_comparison(candidates, "motion_scale"),
            "contraction": _group_comparison(
                candidates,
                "contraction_fraction",
            ),
            "rig": _group_comparison(candidates, "rig_variant"),
            "morphology": _group_comparison(
                candidates,
                "morphology_variant",
            ),
        },
    }
    selected = select_candidate(report)
    report["selected_candidate"] = selected
    report["selection_rationale"] = (
        "highest eligible motion_scale, then highest feasible_fraction, "
        "then lowest max_keypoint_error_m, then lexical candidate_id"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_candidates(args.candidate_roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
