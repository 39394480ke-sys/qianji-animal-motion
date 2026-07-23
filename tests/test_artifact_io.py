from pathlib import Path

import pytest

from qianji_animal_motion.artifact_io import publish_staged_files


def test_publish_rolls_back_when_a_final_file_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    staged = (staging / "one.json", staging / "two.json")
    final = (output / "one.json", output / "two.json")
    staged[0].write_bytes(b"new one")
    staged[1].write_bytes(b"new two")
    final[0].write_bytes(b"old one")
    final[1].write_bytes(b"old two")
    original_replace = Path.replace

    def lose_second_file(source: Path, target: Path) -> Path:
        result = original_replace(source, target)
        target = Path(target)
        if source == staged[1] and target == final[1]:
            target.unlink()
        return result

    monkeypatch.setattr(Path, "replace", lose_second_file)

    with pytest.raises(RuntimeError, match="missing after publishing"):
        publish_staged_files(
            staged,
            final,
            overwrite=True,
            conflict_hint="use --overwrite",
        )

    assert final[0].read_bytes() == b"old one"
    assert final[1].read_bytes() == b"old two"
