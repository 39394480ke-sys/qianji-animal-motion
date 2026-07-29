# Cat 2D-to-3D Mesh Retargeting

This directory reproduces the current cat acceptance case. It uses the
corrected 272-frame six-point trajectory and the current converted cat GLB,
generates a fresh QianJi abstract VGT and quadruped rig, compares motion scales
`0.10`, `0.25`, and `0.50`, and runs QianJi preview and reachability consumers.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'

ANIMAL_DATA_ROOT=/absolute/path/to/qianji-animal-motion \
QIANJI_ROOT=/absolute/path/to/QianJi \
OUTPUT_ROOT=/absolute/path/to/new/output \
bash experiments/2d_to_3d_cat/run_experiment.sh
```

`OUTPUT_ROOT` must not exist. Generated evidence is intentionally ignored by
Git; only this reproducibility harness is versioned.

The result is a body-relative 2.5D retarget, not observed metric depth. See
`docs/experimental_2d_to_3d_mesh.md` for the current measurements and limits.
