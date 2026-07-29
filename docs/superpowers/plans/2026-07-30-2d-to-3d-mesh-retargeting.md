# Experimental 2D-to-3D Mesh Retargeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the current cat case end to end: transfer its corrected six-point
2D motion onto a QianJi quadruped rig generated from the current cat mesh,
emit QianJi-compatible 3D keypoint motion, and prove the result with previews
and reachability evidence.

**Architecture:** Keep lifting math in a dependency-light core module and orchestration in a separate CLI module. The adapter consumes the existing 2D schema plus QianJi `robot.json` and `rig_keypoints.json`, preserves the neutral rig's unobserved lateral coordinate, and publishes motion, binding, and assumption report as one atomic artifact set.

**Tech Stack:** Python 3.12, NumPy, existing artifact publishing helper, pytest, QianJi `biomimic` environment for integration validation.

## Global Constraints

- The current cat is the acceptance case. General animals, non-quadrupeds, and
  full skeleton or mesh deformation are out of scope.
- Reusable code only needs to support the existing six-role quadruped contract.
- The output is `body_relative_2_5d_retarget`, not metric monocular reconstruction.
- `metric_depth_observed`, `camera_calibrated`, and `global_translation_preserved` are always `false`.
- No temporal interpolation, smoothing, or inferred lateral motion is allowed.
- Invalid inputs emit neutral coordinates with confidence `0.0` and an explicit report record.
- Owned outputs are staged and published atomically; overwrite is refused.
- The six semantic role names must match the existing QianJi contracts exactly.

---

### Task 1: Core Body-Relative Lift

**Files:**
- Create: `src/qianji_animal_motion/lift_3d.py`
- Create: `tests/test_lift_3d.py`

**Interfaces:**
- Consumes: trajectory, robot, and rig dictionaries.
- Produces: `LiftResult(motion: dict, binding: dict, report: dict)` from `lift_trajectory(...)`.
- Produces: deterministic `select_reference_frame(...)` and `neutral_pose_from_rig(...)`.

- [ ] **Step 1: Write failing tests for neutral-pose identity and camera-motion removal**

Create synthetic six-role trajectory/robot/rig fixtures. Assert that the
reference frame maps to the neutral pose and that applying image translation,
in-plane rotation, and uniform scale to a frame does not change lifted
coordinates.

- [ ] **Step 2: Run the focused tests and verify missing-module failure**

Run:

```bash
uv run --python 3.12 --with '.[dev]' \
  python -m pytest tests/test_lift_3d.py -q
```

Expected: collection fails because `qianji_animal_motion.lift_3d` does not
exist.

- [ ] **Step 3: Implement validation, neutral rig extraction, and local frames**

Implement:

```python
@dataclass(frozen=True)
class LiftResult:
    motion: dict
    binding: dict
    report: dict

def neutral_pose_from_rig(robot: dict, rig: dict) -> dict[str, np.ndarray]: ...
def select_reference_frame(trajectory: dict) -> int: ...
def lift_trajectory(
    trajectory: dict,
    robot: dict,
    rig: dict,
    *,
    reference_frame: int | None = None,
    motion_scale: float = 0.25,
) -> LiftResult: ...
```

Validate contiguous frames, finite video metadata, six distinct rig sites,
finite 3D site positions, boolean 2D validity, finite valid coordinates, and
positive reference torso length.

- [ ] **Step 4: Verify the focused tests pass**

Run the focused command from Step 2. Expected: all current lift tests pass.

- [ ] **Step 5: Add failing tests for foot lift, lateral preservation, and invalid substitution**

Assert that image-up foot displacement maps along neutral-frame up, every
role's lateral projection remains equal to neutral, an invalid foot becomes
neutral with confidence zero, and an invalid-spine frame becomes a complete
neutral zero-confidence frame.

- [ ] **Step 6: Run new tests and verify behavioral failures**

Run the focused command. Expected: failures identify missing lift and invalid
substitution behavior rather than fixture or import errors.

- [ ] **Step 7: Implement lift equation, confidence handling, and report statistics**

Emit `qianji-keypoint-trajectory-v1`, mesh binding provenance, per-frame
substitution records, basis vectors, displacement ranges, and the four
required false/assumption flags.

- [ ] **Step 8: Run focused and full tests**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest tests/test_lift_3d.py -q
uv run --python 3.12 --with '.[dev]' python -m pytest -q
```

- [ ] **Step 9: Commit core behavior**

```bash
git add src/qianji_animal_motion/lift_3d.py tests/test_lift_3d.py
git commit -m "feat: add body-relative 2d to 3d retargeting"
```

---

### Task 2: Atomic CLI And Package Entry Point

**Files:**
- Create: `src/qianji_animal_motion/lift_cli.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_lift_3d.py`
- Modify: `tests/test_package.py`

**Interfaces:**
- Consumes: three JSON paths and an output directory.
- Produces: `keypoint_motion.json`, `mesh_binding.json`, and
  `lift_report.json`.
- Adds console command `qianji-lift-keypoints`.

- [ ] **Step 1: Write failing CLI tests**

Test one successful run, refusal to overwrite, missing source rejection, source
hash provenance, and preservation of an existing complete output set when a
staged JSON write or publish step fails.

- [ ] **Step 2: Run focused tests and verify failures**

```bash
uv run --python 3.12 --with '.[dev]' \
  python -m pytest tests/test_lift_3d.py tests/test_package.py -q
```

Expected: failures for missing `run_lift` and missing packaged command.

- [ ] **Step 3: Implement staged orchestration**

Implement:

```python
def run_lift(
    *,
    trajectory_path: Path,
    robot_path: Path,
    rig_path: Path,
    output_dir: Path,
    reference_frame: int | None = None,
    motion_scale: float = 0.25,
) -> LiftOutputPaths: ...
```

Hash every source before loading, serialize with `allow_nan=False`, re-check
hashes before publishing, and call existing `publish_staged_files(...)` with
`overwrite=False`.

- [ ] **Step 4: Add argparse entry point**

Add:

```toml
qianji-lift-keypoints = "qianji_animal_motion.lift_cli:main"
```

Support `--trajectory`, `--robot-json`, `--rig`, `--output`,
`--reference-frame`, and `--motion-scale`.

- [ ] **Step 5: Run focused and full tests**

Run the two commands from Task 1 Step 8.

- [ ] **Step 6: Commit CLI**

```bash
git add pyproject.toml src/qianji_animal_motion/lift_cli.py \
  tests/test_lift_3d.py tests/test_package.py
git commit -m "feat: package qianji 3d lift command"
```

---

### Task 3: User Documentation And Data Contract

**Files:**
- Modify: `README.md`
- Modify: `docs/data_contract.md`
- Create: `docs/experimental_2d_to_3d_mesh.md`

**Interfaces:**
- Documents the static mesh -> QianJi morphology -> rig -> lifted motion ->
  reachability path and exact reproducible commands.

- [ ] **Step 1: Document the scientific boundary**

State explicitly that the output is a 2.5D retarget, lateral/depth comes from
the neutral mesh rig, and camera/world motion is removed.

- [ ] **Step 2: Document schemas and invalid semantics**

Describe all three outputs, confidence-zero neutral substitutions, source
hashes, basis vectors, reference selection, and motion-scale behavior.

- [ ] **Step 3: Add the new command to README setup and common commands**

Include an end-to-end command example using QianJi-generated `robot.json` and
`rig_keypoints.json`.

- [ ] **Step 4: Check documentation and full tests**

```bash
git diff --check
uv run --python 3.12 --with '.[dev]' python -m pytest -q
```

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/data_contract.md docs/experimental_2d_to_3d_mesh.md
git commit -m "docs: explain experimental mesh retargeting"
```

---

### Task 4: Real Cat Mesh Integration Experiment

**Files:**
- Create: `experiments/2d_to_3d_cat/README.md`
- Create: `experiments/2d_to_3d_cat/run_experiment.sh`
- Create: `experiments/2d_to_3d_cat/summarize_results.py`
- Generated, ignored: `outputs/experiments/2d_to_3d_cat/**`

**Interfaces:**
- Consumes the current corrected cat trajectory and Assimp GLB from the main
  checkout plus the sibling QianJi repository.
- Produces a mesh-derived abstract VGT, auto-rig, three lifted scale variants,
  QianJi previews, reachability reports, and a combined experiment summary.
- This task, rather than generic API coverage alone, is the completion gate for
  the branch.

- [ ] **Step 1: Write the experiment script with explicit paths and preflight checks**

The script must accept `ANIMAL_DATA_ROOT` and `QIANJI_ROOT`, verify the three
source artifacts, use `mamba run -n biomimic` for QianJi commands, and use the
worktree virtual environment for `qianji-lift-keypoints`.

- [ ] **Step 2: Generate the abstract VGT and auto-rig**

Run QianJi `morph_generator/generate.py 3d-mesh --preset abstract`, then
`controller/check_keypoint_reachability.py --auto-rig quadruped_bbox` to
materialize `rig_keypoints.json`.

- [ ] **Step 3: Lift scales 0.10, 0.25, and 0.50**

Generate isolated output directories and validate every motion JSON with
QianJi's loader.

- [ ] **Step 4: Render previews and run reachability**

For each scale, run `controller/visualize_keypoint_motion.py` and
`controller/check_keypoint_reachability.py` with the generated robot and rig.

- [ ] **Step 5: Summarize quantitative evidence**

Write `experiment_summary.json` and `experiment_summary.md` containing source
SHA-256 values, morphology counts, scale, status counts, feasible fraction,
maximum/mean keypoint error, maximum edge violation, and clipping.

- [ ] **Step 6: Select the example scale without overstating accuracy**

Select the highest scale whose marginal-or-feasible fraction is `1.0`; break
ties by lower maximum keypoint error. If none meet it, select the lowest maximum
error and record that no fully non-unreachable scale was found.

- [ ] **Step 7: Commit reproducibility files only**

```bash
git add experiments/2d_to_3d_cat
git commit -m "exp: add cat mesh retargeting study"
```

---

### Task 5: Final Verification And Branch Record

**Files:**
- Modify: `docs/superpowers/plans/2026-07-30-2d-to-3d-mesh-retargeting.md`
- Generated, ignored: experiment evidence.

**Interfaces:**
- Produces final test, command, artifact, and git-state evidence.

- [ ] **Step 1: Run full test suite**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest -q
```

- [ ] **Step 2: Validate installed CLI and JSON outputs**

```bash
uv run --python 3.12 --with '.[dev]' qianji-lift-keypoints --help
python -m json.tool outputs/experiments/2d_to_3d_cat/experiment_summary.json
```

- [ ] **Step 3: Verify QianJi consumers**

Load the selected motion with QianJi's preview loader, inspect the selected
reachability report, and confirm preview image/video files are non-empty.

- [ ] **Step 4: Audit requirements and repository state**

```bash
git diff --check
git status --short --branch
git log --oneline --decorate -8
```

Confirm every explicit design requirement has corresponding code, tests, docs,
or experiment evidence.

- [ ] **Step 5: Mark the plan and report exact remaining limitations**

Check completed boxes only after evidence exists. Record that camera depth,
camera calibration, global translation, biological joint reconstruction, and
full mesh deformation remain out of scope.
