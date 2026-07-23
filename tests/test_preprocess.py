import json
import shutil
import stat
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
    assert output.stat().st_flags & stat.UF_HIDDEN == 0
    sidecar = output.with_suffix(".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["settings"]["fps"] == 30
    assert payload["output"]["height"] == 720
