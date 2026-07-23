import json
import shutil
import subprocess
from pathlib import Path

import pytest

from qianji_animal_motion.preprocess import (
    VideoMetadata,
    _temporary_output_path,
    build_ffmpeg_command,
    preprocess_video,
    validate_metadata,
)


def test_build_ffmpeg_command_uses_required_normalization(tmp_path: Path) -> None:
    source = tmp_path / "input.mov"
    output = tmp_path / "output.mp4"

    command = build_ffmpeg_command(source, output)

    assert command[:3] == ["ffmpeg", "-hide_banner", "-loglevel"]
    assert "fps=30,scale=-2:720" in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert "-an" in command
    assert command[-1] == str(output)


def test_temporary_output_is_not_hidden(tmp_path: Path) -> None:
    output = tmp_path / "processed.mp4"

    temporary = _temporary_output_path(output)

    assert not temporary.name.startswith(".")
    assert temporary.parent == output.parent
    assert temporary.suffix == ".mp4"


def test_validate_metadata_reports_nonconforming_fields() -> None:
    metadata = VideoMetadata(
        codec="hevc",
        width=1281,
        height=1080,
        fps=29.0,
        pixel_format="yuv444p",
        duration_s=1.0,
        has_audio=True,
    )

    errors = validate_metadata(metadata, expected_fps=30, expected_height=720)

    assert errors == [
        "codec must be h264, got hevc",
        "height must be 720, got 1080",
        "width must be even, got 1281",
        "frame rate must be 30 fps, got 29.000000",
        "pixel format must be yuv420p, got yuv444p",
        "audio stream must be absent",
    ]


def test_validate_metadata_rejects_nonpositive_width() -> None:
    metadata = VideoMetadata(
        codec="h264",
        width=0,
        height=720,
        fps=30.0,
        pixel_format="yuv420p",
        duration_s=1.0,
        has_audio=False,
    )

    assert validate_metadata(metadata) == ["width must be positive, got 0"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are required for the integration test",
)
def test_preprocess_video_normalizes_real_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mov"
    output = tmp_path / "processed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=24:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=0.5",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )

    result = preprocess_video(source, output)

    assert result.codec == "h264"
    assert result.height == 720
    assert result.width == 960
    assert result.fps == pytest.approx(30.0)
    assert result.pixel_format == "yuv420p"
    assert result.has_audio is False
    assert not output.name.startswith(".")
    sidecar = output.with_suffix(".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["settings"]["fps"] == 30
    assert payload["output"]["height"] == 720


def test_preprocess_video_requires_mp4_output(tmp_path: Path) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"video")

    with pytest.raises(ValueError, match=r"\.mp4"):
        preprocess_video(source, tmp_path / "processed.avi")


def test_preprocess_video_reports_ffmpeg_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"video")
    output = tmp_path / "processed.mp4"
    metadata = VideoMetadata(
        codec="h264",
        width=960,
        height=720,
        fps=30.0,
        pixel_format="yuv420p",
        duration_s=1.0,
        has_audio=False,
    )
    monkeypatch.setattr(
        "qianji_animal_motion.preprocess.probe_video",
        lambda *_args, **_kwargs: metadata,
    )

    def fail_ffmpeg(command: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="encoder failed",
        )

    monkeypatch.setattr(
        "qianji_animal_motion.preprocess.subprocess.run",
        fail_ffmpeg,
    )

    with pytest.raises(RuntimeError, match="FFmpeg conversion failed: encoder failed"):
        preprocess_video(source, output)

    assert not output.exists()


def test_preprocess_video_preserves_existing_pair_when_sidecar_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"video")
    output = tmp_path / "processed.mp4"
    sidecar = output.with_suffix(".metadata.json")
    output.write_bytes(b"old video")
    sidecar.write_bytes(b"old metadata")
    metadata = VideoMetadata(
        codec="h264",
        width=960,
        height=720,
        fps=30.0,
        pixel_format="yuv420p",
        duration_s=1.0,
        has_audio=False,
    )

    monkeypatch.setattr(
        "qianji_animal_motion.preprocess.probe_video",
        lambda *_args, **_kwargs: metadata,
    )

    def fake_run(command: list[str], **_kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"new video")

    monkeypatch.setattr(
        "qianji_animal_motion.preprocess.subprocess.run",
        fake_run,
    )
    original_write_text = Path.write_text

    def fail_sidecar(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == sidecar.name:
            raise OSError("sidecar write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_sidecar)

    with pytest.raises(OSError, match="sidecar write failed"):
        preprocess_video(source, output, overwrite=True)

    assert output.read_bytes() == b"old video"
    assert sidecar.read_bytes() == b"old metadata"
