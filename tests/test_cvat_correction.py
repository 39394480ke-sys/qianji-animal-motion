import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from qianji_animal_motion.cvat_correction import (
    _write_json,
    apply_cvat_corrections,
    build_cvat_xml,
    build_review_queue,
    export_cvat_package,
    import_cvat_package,
)


KEYPOINT_NAMES = (
    "spine_front",
    "spine_rear",
    "front_left_foot",
    "front_right_foot",
    "rear_left_foot",
    "rear_right_foot",
)


def _point(x: float, y: float, *, valid: bool = True, flags=None) -> dict:
    return {
        "x_px": x if valid else None,
        "y_px": y if valid else None,
        "confidence": 0.8,
        "valid": valid,
        "source": "model_point",
        "identity_corrected": False,
        "fallback_used": False,
        "flags": list(flags or []),
    }


def _trajectory() -> dict:
    frames = []
    for frame_idx in range(3):
        points = {
            name: _point(10.0 * index + frame_idx, 20.0 + frame_idx)
            for index, name in enumerate(KEYPOINT_NAMES, start=1)
        }
        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / 30.0,
                "keypoints": points,
            }
        )
    frames[1]["keypoints"]["rear_left_foot"] = _point(
        0.0,
        0.0,
        valid=False,
        flags=["identity_ambiguous"],
    )
    return {
        "schema": "qianji.keypoint_trajectory_2d",
        "schema_version": "1.1.0",
        "coordinate_system": "image_pixels_top_left_origin_x_right_y_down",
        "video": {"width": 120, "height": 100, "fps": 30.0, "frame_count": 3},
        "mapping": {},
        "source": {"predictions_sha256": "abc123"},
        "frames": frames,
    }


def _report() -> dict:
    return {
        "identity_corrections": {"front": [], "rear": [1, 2]},
        "identity_ambiguous_frames": {"front": [], "rear": [1]},
        "keypoints": {
            name: {
                "invalid_frames": [1] if name == "rear_left_foot" else [],
                "fallback_used_frames": [],
                "flag_counts": (
                    {"identity_ambiguous": 1}
                    if name == "rear_left_foot"
                    else {}
                ),
            }
            for name in KEYPOINT_NAMES
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xml_root(trajectory: dict) -> ET.Element:
    return ET.fromstring(build_cvat_xml(trajectory))


def _xml_point(root: ET.Element, frame_idx: int, name: str) -> ET.Element:
    skeleton = root.find(f"./track/skeleton[@frame='{frame_idx}']")
    assert skeleton is not None
    point = skeleton.find(f"./points[@label='{name}']")
    assert point is not None
    return point


def _cvat_import_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, dict, str]:
    baseline = _trajectory()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video fixture")
    manifest = {
        "schema": "qianji.cvat_correction_manifest",
        "schema_version": "1.0.0",
        "baseline_trajectory_sha256": _sha256(baseline_path),
        "video_sha256": _sha256(video_path),
    }
    return baseline_path, video_path, manifest, build_cvat_xml(baseline)


def test_cvat_export_writes_every_frame_as_an_explicit_keyframe() -> None:
    root = _xml_root(_trajectory())

    skeletons = root.findall("./track/skeleton")
    assert [item.attrib["frame"] for item in skeletons] == ["0", "1", "2"]
    assert all(item.attrib["keyframe"] == "1" for item in skeletons)
    assert all(
        point.attrib["keyframe"] == "1"
        for skeleton in skeletons
        for point in skeleton.findall("./points")
    )
    assert all(len(item.findall("./points")) == 6 for item in skeletons)


def test_cvat_export_marks_invalid_points_outside_without_inventing_coordinates() -> None:
    point = _xml_point(_xml_root(_trajectory()), 1, "rear_left_foot")

    assert point.attrib["outside"] == "1"
    assert point.attrib["points"] == "0.000000,0.000000"


def test_unedited_cvat_round_trip_keeps_trajectory_unchanged() -> None:
    baseline = _trajectory()
    corrected, corrections, report = apply_cvat_corrections(
        baseline,
        build_cvat_xml(baseline),
    )

    assert corrected == baseline
    assert corrections["corrections"] == []
    assert report["changed_points"] == 0


def test_cvat_coordinate_rounding_is_not_recorded_as_a_manual_move() -> None:
    baseline = _trajectory()
    baseline["frames"][0]["keypoints"]["spine_front"].update(
        x_px=10.0049,
        y_px=20.0049,
    )
    root = _xml_root(baseline)
    point = _xml_point(root, 0, "spine_front")
    point.attrib["points"] = "10.000000,20.000000"

    corrected, corrections, report = apply_cvat_corrections(
        baseline,
        ET.tostring(root, encoding="unicode"),
    )

    assert corrected == baseline
    assert corrections["corrections"] == []
    assert report["changed_points"] == 0


def test_cvat_json_writer_rejects_non_finite_values(tmp_path: Path) -> None:
    output = tmp_path / "unsafe.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        _write_json(output, {"confidence": float("nan")})

    assert not output.exists()


def test_manual_move_changes_only_one_point_and_preserves_model_confidence() -> None:
    baseline = _trajectory()
    root = _xml_root(baseline)
    point = _xml_point(root, 0, "front_left_foot")
    point.attrib["points"] = "44.500000,55.250000"

    corrected, corrections, report = apply_cvat_corrections(
        baseline,
        ET.tostring(root, encoding="unicode"),
    )

    actual = corrected["frames"][0]["keypoints"]["front_left_foot"]
    assert actual["x_px"] == pytest.approx(44.5)
    assert actual["y_px"] == pytest.approx(55.25)
    assert actual["confidence"] == pytest.approx(0.8)
    assert actual["position_source"] == "manual"
    assert actual["review_status"] == "corrected"
    assert actual["flags"] == ["manually_corrected"]
    assert corrections["corrections"][0]["action"] == "move"
    assert report["changed_points"] == 1
    assert corrected["frames"][0]["keypoints"]["spine_front"] == baseline[
        "frames"
    ][0]["keypoints"]["spine_front"]


def test_manual_recovery_makes_an_invalid_point_valid_without_interpolation() -> None:
    baseline = _trajectory()
    root = _xml_root(baseline)
    point = _xml_point(root, 1, "rear_left_foot")
    point.attrib.update(outside="0", points="61.000000,72.000000")

    corrected, corrections, _ = apply_cvat_corrections(
        baseline,
        ET.tostring(root, encoding="unicode"),
    )

    recovered = corrected["frames"][1]["keypoints"]["rear_left_foot"]
    assert recovered["valid"] is True
    assert recovered["x_px"] == pytest.approx(61.0)
    assert recovered["flags"] == ["manually_corrected"]
    assert recovered["correction"]["original_flags"] == ["identity_ambiguous"]
    assert corrections["corrections"][0]["action"] == "recover"
    assert corrected["frames"][0]["keypoints"]["rear_left_foot"]["x_px"] == 50.0
    assert corrected["frames"][2]["keypoints"]["rear_left_foot"]["x_px"] == 52.0


def test_manual_outside_nulls_a_point_and_records_the_original_position() -> None:
    baseline = _trajectory()
    root = _xml_root(baseline)
    point = _xml_point(root, 2, "front_right_foot")
    point.attrib["outside"] = "1"

    corrected, corrections, _ = apply_cvat_corrections(
        baseline,
        ET.tostring(root, encoding="unicode"),
    )

    missing = corrected["frames"][2]["keypoints"]["front_right_foot"]
    assert missing["x_px"] is None
    assert missing["y_px"] is None
    assert missing["valid"] is False
    assert missing["review_status"] == "unresolvable"
    assert "manual_marked_missing" in missing["flags"]
    assert corrections["corrections"][0]["action"] == "mark_missing"
    assert corrections["corrections"][0]["before"]["x_px"] == 42.0


def test_review_queue_prioritizes_invalid_and_identity_boundary_frames() -> None:
    queue = build_review_queue(_trajectory(), _report(), context_frames=1)

    by_frame = {item["frame_idx"]: item for item in queue["frames"]}
    assert by_frame[1]["priority"] == "critical"
    assert "rear_left_foot:identity_ambiguous" in by_frame[1]["reasons"]
    assert "rear_identity_correction_boundary" in by_frame[2]["reasons"]
    assert by_frame[0]["context_only"] is True


def test_export_package_refuses_to_overwrite_and_preserves_inputs(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    trajectory_path = tmp_path / "trajectory.json"
    report_path = tmp_path / "report.json"
    output = tmp_path / "cvat"
    video.write_bytes(b"video fixture")
    trajectory_path.write_text(json.dumps(_trajectory()), encoding="utf-8")
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    before = {
        video: _sha256(video),
        trajectory_path: _sha256(trajectory_path),
        report_path: _sha256(report_path),
    }

    paths = export_cvat_package(
        video_path=video,
        trajectory_path=trajectory_path,
        report_path=report_path,
        output_dir=output,
        skip_video_probe=True,
    )

    assert paths.annotations.is_file()
    assert paths.manifest.is_file()
    assert paths.review_queue.is_file()
    assert all(_sha256(path) == digest for path, digest in before.items())
    with pytest.raises(FileExistsError):
        export_cvat_package(
            video_path=video,
            trajectory_path=trajectory_path,
            report_path=report_path,
            output_dir=output,
            skip_video_probe=True,
        )


def test_import_rejects_wrong_baseline_hash(tmp_path: Path) -> None:
    baseline = _trajectory()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    manifest = {
        "schema": "qianji.cvat_correction_manifest",
        "baseline_trajectory_sha256": "not-the-real-hash",
    }

    from qianji_animal_motion.cvat_correction import import_cvat_package

    with pytest.raises(ValueError, match="baseline trajectory hash"):
        import_cvat_package(
            baseline_path=baseline_path,
            manifest=manifest,
            annotations_xml=build_cvat_xml(copy.deepcopy(baseline)),
            output_dir=tmp_path / "result",
            render_video=False,
        )


def test_import_rejects_manual_coordinates_outside_the_video() -> None:
    baseline = _trajectory()
    root = _xml_root(baseline)
    point = _xml_point(root, 0, "spine_front")
    point.attrib["points"] = "121.000000,20.000000"

    with pytest.raises(ValueError, match="outside video bounds"):
        apply_cvat_corrections(baseline, ET.tostring(root, encoding="unicode"))


def test_import_rejects_recovered_placeholder_at_zero_zero() -> None:
    baseline = _trajectory()
    root = _xml_root(baseline)
    point = _xml_point(root, 1, "rear_left_foot")
    point.attrib["outside"] = "0"

    with pytest.raises(ValueError, match="placeholder coordinates"):
        apply_cvat_corrections(baseline, ET.tostring(root, encoding="unicode"))


def test_cvat_import_discards_staged_outputs_when_preview_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, video, manifest, annotations = _cvat_import_inputs(tmp_path)
    output_dir = tmp_path / "review"
    output_dir.mkdir()
    unrelated = output_dir / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    def fail_preview(_video: Path, _result: object, output: Path) -> None:
        output.write_bytes(b"partial preview")
        raise RuntimeError("preview failed")

    monkeypatch.setattr(
        "qianji_animal_motion.cvat_correction.render_preview",
        fail_preview,
    )

    with pytest.raises(RuntimeError, match="preview failed"):
        import_cvat_package(
            baseline_path=baseline,
            manifest=manifest,
            annotations_xml=annotations,
            output_dir=output_dir,
            video_path=video,
        )

    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not any((output_dir / name).exists() for name in (
        "keypoint_trajectory_2d_corrected.json",
        "corrections.json",
        "correction_report.json",
        "six_keypoints_corrected_preview.mp4",
    ))
    assert not list(tmp_path.glob(".review.staging-*"))


def test_cvat_import_discards_outputs_when_baseline_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, video, manifest, annotations = _cvat_import_inputs(tmp_path)
    output_dir = tmp_path / "review"

    def mutate_baseline(_video: Path, _result: object, output: Path) -> None:
        baseline.write_text('{"changed": true}', encoding="utf-8")
        output.write_bytes(b"preview")

    monkeypatch.setattr(
        "qianji_animal_motion.cvat_correction.render_preview",
        mutate_baseline,
    )

    with pytest.raises(RuntimeError, match="baseline trajectory changed"):
        import_cvat_package(
            baseline_path=baseline,
            manifest=manifest,
            annotations_xml=annotations,
            output_dir=output_dir,
            video_path=video,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".review.staging-*"))


def test_cvat_import_discards_outputs_when_source_video_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, video, manifest, annotations = _cvat_import_inputs(tmp_path)
    output_dir = tmp_path / "review"

    def mutate_video(_video: Path, _result: object, output: Path) -> None:
        video.write_bytes(b"changed during import")
        output.write_bytes(b"preview")

    monkeypatch.setattr(
        "qianji_animal_motion.cvat_correction.render_preview",
        mutate_video,
    )

    with pytest.raises(RuntimeError, match="source video changed"):
        import_cvat_package(
            baseline_path=baseline,
            manifest=manifest,
            annotations_xml=annotations,
            output_dir=output_dir,
            video_path=video,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".review.staging-*"))


def test_cvat_import_without_preview_publishes_three_json_files(
    tmp_path: Path,
) -> None:
    baseline, _video, manifest, annotations = _cvat_import_inputs(tmp_path)
    output_dir = tmp_path / "review"

    paths = import_cvat_package(
        baseline_path=baseline,
        manifest=manifest,
        annotations_xml=annotations,
        output_dir=output_dir,
        render_video=False,
    )

    assert paths.preview is None
    assert paths.trajectory.is_file()
    assert paths.corrections.is_file()
    assert paths.report.is_file()
    assert not list(tmp_path.glob(".review.staging-*"))


def test_cvat_import_rolls_back_when_publishing_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, video, manifest, annotations = _cvat_import_inputs(tmp_path)
    output_dir = tmp_path / "review"
    original_replace = Path.replace
    publish_count = 0

    def write_preview(_video: Path, _result: object, output: Path) -> None:
        output.write_bytes(b"preview")

    def fail_second_publish(source: Path, target: Path) -> Path:
        nonlocal publish_count
        target = Path(target)
        if (
            source.parent.name.startswith(".review.staging-")
            and target.parent == output_dir
        ):
            publish_count += 1
            if publish_count == 2:
                raise OSError("publish failed")
        return original_replace(source, target)

    monkeypatch.setattr(
        "qianji_animal_motion.cvat_correction.render_preview",
        write_preview,
    )
    monkeypatch.setattr(Path, "replace", fail_second_publish)

    with pytest.raises(OSError, match="publish failed"):
        import_cvat_package(
            baseline_path=baseline,
            manifest=manifest,
            annotations_xml=annotations,
            output_dir=output_dir,
            video_path=video,
        )

    assert not any((output_dir / name).exists() for name in (
        "keypoint_trajectory_2d_corrected.json",
        "corrections.json",
        "correction_report.json",
        "six_keypoints_corrected_preview.mp4",
    ))
    assert not list(tmp_path.glob(".review.staging-*"))
