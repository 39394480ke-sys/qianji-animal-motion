# 39-Point Cat VGT Experiment

This experiment is the reproducible real-data acceptance run for:

```text
cat video + Hunyuan3D mesh
-> literal 39-point SuperAnimal observations
-> body-relative mesh-assisted 2.5D landmarks
-> six explicit structural controls
-> QianJi 12-site/30-rod VGT sequence
-> 30 FPS structure animation
```

It does not reconstruct a new mesh, infer metric camera depth, deform the
2,154 mesh vertices, or claim a MuJoCo dynamics result.

## Run

Install the current worktree into its virtual environment first:

```bash
uv pip install --python .venv/bin/python -e .
```

Then choose a new output directory:

```bash
ANIMAL_DATA_ROOT="/absolute/qianji-animal-motion" \
QIANJI_ROOT="/absolute/QianJi" \
OUTPUT_ROOT="$PWD/outputs/experiments/39point_vgt_cat_v1" \
bash experiments/39point_vgt_cat/run_experiment.sh
```

The script rejects an existing `OUTPUT_ROOT`, treats QianJi as read-only,
hashes the real inputs, runs the bounded six-candidate reachability study,
selects only a zero-unreachable result below 5 cm maximum control error, and
exits nonzero unless `reports/final_acceptance_report.json` passes.

The selected result is copied into the stable `landmarks/`, `control/`,
`motion/`, and `previews/` paths. Raw candidate evidence remains under
`candidates/`.
