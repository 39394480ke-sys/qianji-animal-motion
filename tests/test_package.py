from pathlib import Path
import tomllib

from qianji_animal_motion import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_anchor_suggestion_command_is_packaged() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"]["qianji-suggest-anchor"] == (
        "qianji_animal_motion.anchor_selection:main"
    )
