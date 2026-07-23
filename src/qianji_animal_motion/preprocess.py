"""Normalize source videos before animal pose inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from qianji_animal_motion.artifact_io import publish_staged_files


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
X264_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


@dataclass(frozen=True)
class VideoMetadata:
    codec: str
    width: int
    height: int
    fps: float
    pixel_format: str
    duration_s: float
    has_audio: bool


def _parse_rate(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    if float(denominator) == 0:
        return 0.0
    return float(numerator) / float(denominator)


def probe_video(path: Path, ffprobe: str = "ffprobe") -> VideoMetadata:
    """Read the first video stream and report properties relevant to inference."""
    path = Path(path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffprobe was not found: {ffprobe}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"could not inspect {path}: {detail}") from exc

    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise ValueError(f"no video stream found in {path}")

    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    duration = video.get("duration") or payload.get("format", {}).get("duration") or 0
    return VideoMetadata(
        codec=str(video.get("codec_name", "")),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=_parse_rate(str(rate)),
        pixel_format=str(video.get("pix_fmt", "")),
        duration_s=float(duration),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    *,
    fps: int = 30,
    height: int = 720,
    crf: int = 18,
    preset: str = "medium",
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the deterministic FFmpeg command used by the preprocessor."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps},scale=-2:{height}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def validate_metadata(
    metadata: VideoMetadata,
    *,
    expected_fps: int = 30,
    expected_height: int = 720,
) -> list[str]:
    """Return all normalization violations found in output metadata."""
    errors: list[str] = []
    if metadata.codec != "h264":
        errors.append(f"codec must be h264, got {metadata.codec}")
    if metadata.height != expected_height:
        errors.append(f"height must be {expected_height}, got {metadata.height}")
    if metadata.width <= 0:
        errors.append(f"width must be positive, got {metadata.width}")
    elif metadata.width % 2:
        errors.append(f"width must be even, got {metadata.width}")
    if not math.isclose(metadata.fps, expected_fps, rel_tol=0, abs_tol=1e-3):
        errors.append(
            f"frame rate must be {expected_fps} fps, got {metadata.fps:.6f}"
        )
    if metadata.pixel_format != "yuv420p":
        errors.append(
            f"pixel format must be yuv420p, got {metadata.pixel_format}"
        )
    if metadata.has_audio:
        errors.append("audio stream must be absent")
    return errors


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_video(
    input_path: Path,
    output_path: Path,
    *,
    fps: int = 30,
    height: int = 720,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> VideoMetadata:
    """Normalize one video, validate it, and write a metadata sidecar."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("input and output paths must be different")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"output must use the .mp4 extension: {output_path}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if height <= 0 or height % 2:
        raise ValueError("height must be a positive even number")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    if preset not in X264_PRESETS:
        raise ValueError(f"unsupported x264 preset: {preset}")
    sidecar = output_path.with_suffix(".metadata.json")
    existing = [path for path in (output_path, sidecar) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output already exists: {existing[0]}; use --overwrite to replace it"
        )

    input_hash = _sha256(input_path)
    input_metadata = probe_video(input_path, ffprobe=ffprobe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_path.stem}.staging-",
        dir=output_path.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        staged_output = staging_dir / output_path.name
        staged_sidecar = staging_dir / sidecar.name
        temporary_path = _temporary_output_path(staged_output)
        command = build_ffmpeg_command(
            input_path,
            temporary_path,
            fps=fps,
            height=height,
            crf=crf,
            preset=preset,
            ffmpeg=ffmpeg,
        )
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            output_metadata = probe_video(temporary_path, ffprobe=ffprobe)
            errors = validate_metadata(
                output_metadata,
                expected_fps=fps,
                expected_height=height,
            )
            if errors:
                raise RuntimeError(
                    "output validation failed: " + "; ".join(errors)
                )
            temporary_path.replace(staged_output)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg was not found: {ffmpeg}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (
                (exc.stderr or "").strip()
                or (exc.stdout or "").strip()
                or "unknown error"
            )
            raise RuntimeError(f"FFmpeg conversion failed: {detail}") from exc
        finally:
            temporary_path.unlink(missing_ok=True)

        staged_sidecar.write_text(
            json.dumps(
                {
                    "schema": "qianji.video_preprocess",
                    "schema_version": "1.0.0",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "input_sha256": input_hash,
                    "settings": {
                        "fps": fps,
                        "height": height,
                        "video_codec": "h264",
                        "pixel_format": "yuv420p",
                        "audio_removed": True,
                        "cropped": False,
                        "crf": crf,
                        "preset": preset,
                    },
                    "input": asdict(input_metadata),
                    "output": asdict(output_metadata),
                },
                ensure_ascii=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if _sha256(input_path) != input_hash:
            raise RuntimeError("source video changed during preprocessing")
        publish_staged_files(
            (staged_output, staged_sidecar),
            (output_path, sidecar),
            overwrite=overwrite,
            conflict_hint="use --overwrite to replace it",
        )
    return output_metadata


def _default_output(source: Path, fps: int, height: int) -> Path:
    if source.is_dir():
        return source.parent / "processed"
    return source.with_name(f"{source.stem}_{fps}fps_{height}p.mp4")


def _video_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize videos for DeepLabCut pose inference."
    )
    parser.add_argument("input", type=Path, help="Input video or directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 for one video, or output directory for batch mode",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.input.expanduser().resolve()
    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output(source, args.fps, args.height)
    )

    if source.is_dir():
        videos = _video_files(source)
        if not videos:
            raise SystemExit(f"no supported videos found in {source}")
        destination.mkdir(parents=True, exist_ok=True)
        for video in videos:
            output = destination / f"{video.stem}_{args.fps}fps_{args.height}p.mp4"
            metadata = preprocess_video(
                video,
                output,
                fps=args.fps,
                height=args.height,
                crf=args.crf,
                preset=args.preset,
                overwrite=args.overwrite,
            )
            print(f"processed {video.name} -> {output.name} ({metadata.fps:.3f} fps)")
        return 0

    metadata = preprocess_video(
        source,
        destination,
        fps=args.fps,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
        overwrite=args.overwrite,
    )
    print(f"processed {source.name} -> {destination} ({metadata.fps:.3f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
