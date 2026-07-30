"""Validation and packaging primitives for QianJi VGT model sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VgtSequence:
    site_names: tuple[str, ...]
    times: np.ndarray
    positions: np.ndarray
    rods: tuple[tuple[str, str], ...]


def _validate_rods(
    robot: dict,
    site_names: tuple[str, ...],
    *,
    expected_rods: int,
) -> tuple[tuple[str, str], ...]:
    rods = robot.get("rod_groups")
    if not isinstance(rods, list) or len(rods) != expected_rods:
        raise ValueError(f"robot must contain exactly {expected_rods} rods")
    site_set = set(site_names)
    output = []
    rod_names = set()
    for index, rod in enumerate(rods):
        if not isinstance(rod, dict):
            raise ValueError(f"rod {index} must be an object")
        name = rod.get("name")
        site1 = rod.get("site1")
        site2 = rod.get("site2")
        if not isinstance(name, str) or not name or name in rod_names:
            raise ValueError("rod names must be non-empty and unique")
        rod_names.add(name)
        if site1 not in site_set or site2 not in site_set:
            raise ValueError(f"rod {name!r} references a missing site")
        if site1 == site2:
            raise ValueError(f"rod {name!r} must connect different sites")
        output.append((site1, site2))
    return tuple(output)


def load_and_validate_vgt_sequence(
    npz_path: Path,
    robot: dict,
    *,
    expected_frames: int,
    expected_sites: int,
    expected_rods: int,
) -> VgtSequence:
    """Load finite QianJi site targets with an exact robot topology contract."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"VGT sequence does not exist: {path}")
    if expected_frames <= 0 or expected_sites <= 0 or expected_rods <= 0:
        raise ValueError("expected counts must be positive")
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = {"site_names", "times", "positions"} - set(archive.files)
            if missing:
                raise ValueError(
                    "VGT NPZ is missing arrays: " + ", ".join(sorted(missing))
                )
            raw_names = np.array(archive["site_names"], copy=True)
            times = np.asarray(archive["times"], dtype=float).copy()
            positions = np.asarray(archive["positions"], dtype=float).copy()
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("VGT NPZ"):
            raise
        raise ValueError(f"cannot load VGT NPZ: {path}") from error

    if raw_names.shape != (expected_sites,):
        raise ValueError(
            f"site_names must contain exactly {expected_sites} sites"
        )
    if raw_names.dtype.kind not in {"U", "S"}:
        raise ValueError("site_names must be a string array")
    site_names = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_names.tolist()
    )
    if any(not name for name in site_names) or len(set(site_names)) != len(
        site_names
    ):
        raise ValueError("site_names must be non-empty and unique")

    sites = robot.get("sites")
    if not isinstance(sites, dict) or len(sites) != expected_sites:
        raise ValueError(f"robot must contain exactly {expected_sites} sites")
    if site_names != tuple(sites):
        raise ValueError("site_names must exactly match robot site order")
    if times.shape != (expected_frames,):
        raise ValueError(f"times must contain exactly {expected_frames} frames")
    if positions.shape != (expected_frames, expected_sites, 3):
        raise ValueError(
            "positions shape must be "
            f"({expected_frames}, {expected_sites}, 3) frames/sites/XYZ"
        )
    if not np.isfinite(times).all() or not np.isfinite(positions).all():
        raise ValueError("VGT times and positions must be finite")
    if expected_frames > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("VGT times must be strictly increasing")
    rods = _validate_rods(
        robot,
        site_names,
        expected_rods=expected_rods,
    )
    times.setflags(write=False)
    positions.setflags(write=False)
    return VgtSequence(
        site_names=site_names,
        times=times,
        positions=positions,
        rods=rods,
    )


def validate_render_rate(sequence: VgtSequence, fps: float) -> None:
    """Validate the declared display rate without resampling the sequence."""
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("render fps must be finite and positive")
    if sequence.times.size > 1:
        expected_step = 1.0 / fps
        actual_steps = np.diff(sequence.times)
        if not np.allclose(actual_steps, expected_step, atol=1e-6, rtol=1e-5):
            raise ValueError("VGT times do not match the requested render fps")
