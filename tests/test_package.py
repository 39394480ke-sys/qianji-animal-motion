from pathlib import Path
import tomllib

from qianji_animal_motion import __version__


def test_package_version() -> None:
    assert __version__ == "0.2.0"


def test_all_pipeline_commands_are_packaged() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["scripts"] == {
        "qianji-convert-mesh": "qianji_animal_motion.mesh_convert:main",
        "qianji-preprocess-video": "qianji_animal_motion.preprocess:main",
        "qianji-suggest-anchor": "qianji_animal_motion.anchor_selection:main",
        "qianji-map-keypoints": "qianji_animal_motion.semantic_cli:main",
        "qianji-export-cvat": "qianji_animal_motion.cvat_correction:export_main",
        "qianji-import-cvat": "qianji_animal_motion.cvat_correction:import_main",
        "qianji-lift-keypoints": "qianji_animal_motion.lift_cli:main",
        "qianji-export-39-keypoints": (
            "qianji_animal_motion.keypoints_39_cli:main"
        ),
    }
