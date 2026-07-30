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

    motion_scale = _finite_number(parameters.get("motion_scale"), "motion scale")
    contraction = _finite_number(
        parameters.get("contraction_fraction"),
        "contraction fraction",
    )
    rig_variant = parameters.get("rig_variant")
    morphology = parameters.get("morphology_variant")
    if not isinstance(rig_variant, str) or not isinstance(morphology, str):
        raise ValueError("candidate rig and morphology variants must be strings")
    failures = []
    if frames != int(metadata.get("expected_frames", frames)):
        failures.append("frame_count")
    if unreachable != 0:
        failures.append("unreachable_frames")
    if max_error > 0.05:
        failures.append("max_keypoint_error_above_0.05_m")
    if sequence.positions.shape != (frames, 12, 3):
        failures.append("vgt_shape")
    hashes = {label: _sha256(path) for label, path in paths.items()}
    if hashes["desired_motion"] == hashes["projected_motion"]:
        failures.append("desired_projected_content_alias")
    record = {
        "candidate_id": metadata.get("candidate_id", root.name),
        "candidate_root": str(root),
        "motion_scale": motion_scale,
        "contraction_fraction": contraction,
        "rig_variant": rig_variant,
        "morphology_variant": morphology,
        "summary": summary,
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


def compare_candidates(candidate_roots: list[Path]) -> dict:
    """Validate all artifacts, aggregate parameters, and select one candidate."""
    if not candidate_roots:
        raise ValueError("at least one candidate root is required")
    candidates = [_candidate_record(Path(root)) for root in candidate_roots]
    ids = [item["candidate_id"] for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    report = {
        "schema": "qianji.39point_reachability_comparison",
        "schema_version": "0.1.0",
        "acceptance": {
            "unreachable_frames": 0,
            "max_keypoint_error_m": 0.05,
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
