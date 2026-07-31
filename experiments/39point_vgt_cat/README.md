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
verifies the source Mesh against a frozen QianJi-generated 12-site/30-rod
template, and builds the whole case in a hidden staging directory. It runs a
seven-candidate reachability study and only publishes a candidate with zero
unreachable frames, at most 5 cm control error, at most 0.5 mm rod violation,
and at most 5% clipped or independently violated rods.

The selected result is copied into the stable `landmarks/`, `control/`,
`motion/`, and `previews/` paths. Raw candidate evidence remains under
`candidates/`. The final verifier replays the original H5 through observation,
2.5D lifting, control mapping, and desired controls. It independently derives
control errors, frame statuses, feasible fractions, NPZ rod limits, and checks
the selected XML slide axes, actuators, and weld topology. Formal publication
also requires unchanged clean repository, QianJi, tool, input, project-Python,
and QianJi/Mamba-Python provenance.
Automated acceptance does not replace human visual review of a newly generated
version.
