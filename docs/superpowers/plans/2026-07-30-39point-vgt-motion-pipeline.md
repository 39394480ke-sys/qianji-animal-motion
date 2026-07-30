# 39-Point VGT Motion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the reproducible 272-frame cat pipeline from the real
SuperAnimal 39-point H5 and Hunyuan3D-derived mesh to 39-point body-relative
2.5D motion, a selected 12-site/30-rod QianJi VGT sequence, and 30 FPS
structure animation.

**Architecture:** Preserve the existing six-point modules and add four focused
layers: 39-point observation export, neutral-landmark/lift, VGT control
adaptation, and VGT sequence packaging/rendering. A versioned experiment
script treats QianJi as a read-only command dependency, evaluates explicit
reachability candidates, selects an eligible result, and publishes a
machine-verifiable case directory.

**Tech Stack:** Python 3.12, NumPy, pandas/PyTables, OpenCV, existing artifact
publisher, pytest, QianJi `biomimic` Mamba environment, FFmpeg/FFprobe.

## Global Constraints

- Branch: `experiment/39point-vgt-motion-pipeline`.
- Base: `experiment/2d-to-3d-mesh-mapping`.
- The existing six-point public behavior and tests remain unchanged.
- The real video has exactly 272 frames at 30 FPS and 1246x720.
- The observation layer contains exactly the 39 source H5 bodyparts.
- No temporal interpolation or smoothing is permitted.
- Invalid roles remain present with explicit validity/confidence semantics.
- Three-dimensional output is `body_relative_2_5d_retarget`, not observed
  camera depth, world translation, or metric reconstruction.
- QianJi is read-only; all modified robot constraints are copied artifacts.
- Source and generated artifacts are never overwritten.
- JSON writes use `allow_nan=False`; NPZ arrays must be finite.
- The selected VGT positions shape is exactly `(272, 12, 3)`.
- All 30 rods must reference existing sites.
- Selected reachability has zero unreachable frames and maximum control-point
  error no greater than `0.05` m.
- No 2,154-vertex mesh deformation and no MuJoCo dynamics are implemented.
- Every task ends with focused tests, the full suite, and one clear commit.

---

### Task 1: 39-Point Observation Core

**Files:**
- Create: `src/qianji_animal_motion/keypoints_39.py`
- Create: `tests/test_keypoints_39.py`

**Interfaces:**
- Consumes: a SuperAnimal pandas DataFrame, `VideoInfo`, individual name,
  confidence threshold, and frame-152 front/rear anchor states.
- Produces:

```python
SUPERANIMAL_QUADRUPED_39: tuple[str, ...]

@dataclass(frozen=True)
class Observation39Result:
    trajectory: dict
    report: dict

def build_39point_observation(
    dataframe: pd.DataFrame,
    video: VideoInfo,
    *,
    individual: str = "animal0",
    confidence_threshold: float = 0.3,
    anchor_frame: int = 152,
    front_anchor: str = "keep",
    rear_anchor: str = "keep",
) -> Observation39Result: ...
```

- [ ] **Step 1: Write failing tests for exact role and frame preservation**

Build a two-frame real pandas MultiIndex fixture containing all literal
`SUPERANIMAL_QUADRUPED_39` roles and three coordinates. Assert:

```python
assert trajectory["schema"] == "qianji.keypoint_trajectory_2d_39"
assert len(trajectory["frames"]) == 2
assert tuple(trajectory["frames"][0]["keypoints"]) == SUPERANIMAL_QUADRUPED_39
assert all(len(frame["keypoints"]) == 39 for frame in trajectory["frames"])
```

The production change caught is dropping, reordering, or renaming an H5 role.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
uv run --python 3.12 --with '.[dev]' \
  python -m pytest tests/test_keypoints_39.py -q
```

Expected: collection fails because `keypoints_39` does not exist.

- [ ] **Step 3: Implement schema validation and literal point export**

Validate one scorer, one requested individual, exactly the 39 expected
bodyparts, `x/y/likelihood`, contiguous frame index, positive video metadata,
finite source values, and threshold in `[0,1]`. Emit every role in every frame
without interpolation.

- [ ] **Step 4: Add failing quality-policy tests**

Use literal points to assert:

- confidence `0.29` at threshold `0.30` becomes `valid: false`, null `x_px` and
  `y_px`, and flag `low_confidence`;
- a point outside 100x80 becomes invalid with `out_of_bounds`;
- an abrupt displacement greater than `0.75 * torso_scale` becomes invalid
  with `temporal_jump`;
- the report contains literal invalid frame indices and flag counts.

The production changes caught are accepting bad points, silently clipping
coordinates, or omitting quality evidence.

- [ ] **Step 5: Implement deterministic quality classification**

Use finite raw coordinates for quality computation. Estimate per-frame torso
scale from `back_end` to `back_base`, falling back only to the video diagonal
for threshold calculation and recording that fallback. A temporal jump is
compared only with the previous raw frame and never repaired.

- [ ] **Step 6: Add failing complete-chain identity tests**

Construct frames where the resolved front identity swaps. Assert that
`front_left_thai`, `front_left_knee`, and `front_left_paw` all receive the raw
right-chain values, and the report records the frame once. Repeat for the rear
chain. The production change caught is swapping only a paw or mixing joints
from different legs.

- [ ] **Step 7: Implement existing resolver integration**

Call `resolve_leg_identities(...)` independently for the complete front and
rear three-joint chains. Apply the chosen state before quality classification.
Record states, ambiguous frames, and confirmed anchor semantics.

- [ ] **Step 8: Run focused and full tests**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest tests/test_keypoints_39.py -q
uv run --python 3.12 --with '.[dev]' python -m pytest -q
```

- [ ] **Step 9: Commit the observation core**

```bash
git add src/qianji_animal_motion/keypoints_39.py tests/test_keypoints_39.py
git commit -m "feat: export complete 39-point observations"
```

---

### Task 2: Observation CLI, Input Manifest, and 2D Preview

**Files:**
- Create: `src/qianji_animal_motion/keypoints_39_cli.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_keypoints_39.py`
- Modify: `tests/test_package.py`

**Interfaces:**
- Produces `Observation39Paths` containing:
  `input_manifest.json`, `keypoint_trajectory_2d_39.json`,
  `keypoint_39_quality_report.json`, and `keypoint_39_preview.mp4`.
- Adds console command:

```toml
qianji-export-39-keypoints = "qianji_animal_motion.keypoints_39_cli:main"
```

- [ ] **Step 1: Write a failing real-file orchestration test**

Create a two-frame H5 with all 39 roles and a two-frame 100x80 MP4. Create
small mesh and corrected-spine JSON files. Call:

```python
run_observation_export(
    video_path=video,
    predictions_path=h5,
    mesh_path=mesh,
    corrected_spine_path=spine,
    output_dir=tmp_path / "case",
    reference_frame=1,
    mesh_generation_method="hunyuan3d_from_video_frame",
)
```

Assert four complete outputs, exact source SHA-256 values, declared Hunyuan3D
provenance, two preview frames, and refusal to overwrite.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: import failure for `keypoints_39_cli`.

- [ ] **Step 3: Implement source hashing, staged publication, and manifest**

The manifest includes absolute paths and SHA-256 for video, H5, mesh, corrected
spine, plus reference frame, video metadata, H5 scorer/individual/bodyparts,
mesh generation method, and the four false scientific flags. Re-hash all
sources immediately before atomic publication.

- [ ] **Step 4: Add a failing preview test**

Open the published MP4 with OpenCV and assert width 100, height 80, FPS 30,
two frames, and non-background colored pixels at a known valid point. The
production change caught is writing an empty, wrong-rate, or unannotated file.

- [ ] **Step 5: Implement 39-point preview rendering**

Draw valid points by anatomical group, draw only edges whose endpoints are
valid, render a compact invalid-count overlay, and preserve the source frame
size and FPS. Check that the writer emitted exactly the trajectory frame
count.

- [ ] **Step 6: Add CLI arguments and package test**

Support `--video`, `--predictions`, `--mesh`, `--corrected-spine`, `--output`,
`--reference-frame`, `--individual`, `--confidence-threshold`, and
`--mesh-generation-method`.

- [ ] **Step 7: Run focused/full tests and commit**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest \
  tests/test_keypoints_39.py tests/test_package.py -q
uv run --python 3.12 --with '.[dev]' python -m pytest -q
git add pyproject.toml src/qianji_animal_motion/keypoints_39_cli.py \
  tests/test_keypoints_39.py tests/test_package.py
git commit -m "feat: package 39-point observation artifacts"
```

---

### Task 3: Neutral 39 Landmarks and 2.5D Lift

**Files:**
- Create: `src/qianji_animal_motion/lift_39.py`
- Create: `src/qianji_animal_motion/lift_39_cli.py`
- Create: `tests/test_lift_39.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class Lift39Result:
    neutral_landmarks: dict
    motion: dict
    report: dict

def build_neutral_landmarks_39(
    trajectory_39: dict,
    corrected_spine: dict,
    robot: dict,
    rig: dict,
    *,
    reference_frame: int,
) -> dict: ...

def lift_39point_trajectory(
    trajectory_39: dict,
    corrected_spine: dict,
    neutral_landmarks: dict,
    *,
    motion_scale: float,
) -> Lift39Result: ...
```

- [ ] **Step 1: Write failing neutral-frame tests**

Use a literal six-site rig embedded in a 12-site robot and 39 reference points.
Assert:

- all 39 neutral roles exist;
- left/right paired roles have opposite nonzero lateral projections;
- centerline roles have zero lateral projection;
- basis vectors are finite and orthonormal;
- antler roles are present with `anatomy_applicable: false`;
- source hashes and reference confidence fields are represented.

- [ ] **Step 2: Run and verify RED**

Expected: missing `lift_39` import.

- [ ] **Step 3: Implement neutral construction**

Reuse `neutral_pose_from_rig` for the six rig roles and the existing 3D basis
convention. Derive each landmark's reference longitudinal/vertical coordinate
from the raw finite frame-152 H5 position and corrected spine body frame.
Derive lateral sign from semantic side and lateral magnitude from the rig
left/right width.

- [ ] **Step 4: Add failing lift tests**

Assert:

- reference frame equals neutral XYZ for every role;
- translation, in-plane rotation, and uniform zoom do not change output;
- an image-up knee movement maps along positive 3D up;
- lateral projection never changes;
- an invalid role emits neutral XYZ and confidence zero;
- every frame contains all 39 roles;
- report substitutions are explicit and no interpolation field exists.

- [ ] **Step 5: Implement 39-point body-relative lift**

Use the corrected six-point spine per frame for the body frame and the same
equation as the approved design. Reject missing/invalid spine frames rather
than borrowing from another time. Serialize as:

```text
schema: qianji-keypoint-trajectory-39-v1
reconstruction_kind: body_relative_2_5d_retarget
```

- [ ] **Step 6: Write failing atomic CLI tests**

Test source-hash provenance, non-overwrite, malformed role rejection, source
mutation rejection, `allow_nan=False`, and complete publication of
`neutral_landmarks_39.json`, `keypoint_motion_3d_39.json`, and
`lift_39_report.json`.

- [ ] **Step 7: Implement and package `qianji-lift-39-keypoints`**

Arguments: `--trajectory-39`, `--corrected-spine`, `--robot-json`, `--rig`,
`--output`, `--reference-frame`, and `--motion-scale`.

- [ ] **Step 8: Run tests and commit**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest tests/test_lift_39.py -q
uv run --python 3.12 --with '.[dev]' python -m pytest -q
git add pyproject.toml src/qianji_animal_motion/lift_39.py \
  src/qianji_animal_motion/lift_39_cli.py tests/test_lift_39.py \
  tests/test_package.py
git commit -m "feat: lift 39 landmarks onto mesh-derived VGT frame"
```

---

### Task 4: VGT Control Mapping and Candidate Preparation

**Files:**
- Create: `src/qianji_animal_motion/vgt_control.py`
- Create: `src/qianji_animal_motion/vgt_control_cli.py`
- Create: `tests/test_vgt_control.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package.py`

**Interfaces:**

```python
def build_vgt_control_map(
    robot: dict,
    rig: dict,
    neutral_landmarks: dict,
) -> dict: ...

def build_target_control_motion(
    motion_39: dict,
    neutral_landmarks: dict,
    robot: dict,
    control_map: dict,
) -> tuple[dict, dict]: ...

def build_motion_informed_rig(
    robot: dict,
    neutral_landmarks: dict,
) -> dict: ...

def apply_contraction_range(robot: dict, fraction: float) -> dict: ...
```

- [ ] **Step 1: Write failing reference and fallback tests**

Assert that each control role names its observation primary/fallback and rig
site, all six sites are distinct, the reference target equals the rig neutral
XYZ, and an invalid primary uses only its named same-frame fallback while
recording it.

- [ ] **Step 2: Run and verify RED**

Expected: missing `vgt_control` import.

- [ ] **Step 3: Implement the explicit observation/control boundary**

Use the six mappings from the design. Apply observation displacement relative
to neutral landmark to the mapped site neutral. Emit
`qianji-keypoint-trajectory-v1` for QianJi and a report with every fallback or
neutral substitution.

- [ ] **Step 4: Add failing motion-informed rig tests**

With literal site positions, assert nearest semantic assignment selects six
distinct existing sites and rejects robots with fewer than six sites. The
expected site IDs are hand-derived in the fixture.

- [ ] **Step 5: Implement deterministic distinct-site assignment**

Build a role-to-site distance matrix and choose the minimum total distinct
assignment using exhaustive permutations over the nearest bounded candidates
for the six roles. Tie-break lexicographically by site tuple.

- [ ] **Step 6: Add failing contraction tests**

For fraction `0.10`, assert every copied rod's
`effective_min_length == 0.9 * effective_current_length`, maximum and current
length are unchanged, mode names the experiment, the source robot is
unchanged, and fractions outside `[0,0.5]` fail.

- [ ] **Step 7: Implement candidate CLI**

`qianji-prepare-vgt-control` writes versioned bbox/motion-informed rigs,
control maps, target motions, and contraction-adjusted robot copies. It never
modifies QianJi or input files.

- [ ] **Step 8: Run tests and commit**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest tests/test_vgt_control.py -q
uv run --python 3.12 --with '.[dev]' python -m pytest -q
git add pyproject.toml src/qianji_animal_motion/vgt_control.py \
  src/qianji_animal_motion/vgt_control_cli.py tests/test_vgt_control.py \
  tests/test_package.py
git commit -m "feat: map 39-point motion to VGT controls"
```

---

### Task 5: VGT Sequence Contract and 12-Site/30-Rod Renderer

**Files:**
- Create: `src/qianji_animal_motion/vgt_sequence.py`
- Create: `src/qianji_animal_motion/vgt_render.py`
- Create: `src/qianji_animal_motion/vgt_sequence_cli.py`
- Create: `tests/test_vgt_sequence.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class VgtSequence:
    site_names: tuple[str, ...]
    times: np.ndarray
    positions: np.ndarray

def load_and_validate_vgt_sequence(
    npz_path: Path,
    robot: dict,
    *,
    expected_frames: int,
    expected_sites: int,
    expected_rods: int,
) -> VgtSequence: ...

def package_vgt_sequence(...) -> dict: ...
def render_vgt_motion(...) -> dict: ...
```

- [ ] **Step 1: Write failing NPZ and rod-contract tests**

Use three frames, 12 sites, 30 literal valid rods. Assert accepted shape
`(3,12,3)`, strictly increasing times, exact site order, and 30 endpoint
pairs. Independently test rejection of NaN, duplicate site names, missing rod
endpoints, wrong frame/site dimensions, and non-monotonic time.

- [ ] **Step 2: Run and verify RED**

Expected: missing `vgt_sequence` import.

- [ ] **Step 3: Implement validated packaging**

Copy arrays by value, write compressed NPZ, hash it, and emit a manifest
containing selected candidate, exact shapes, time range, FPS, sites, rods,
robot/rig hashes, desired target path/hash, projected path/hash, and the four
scientific false flags.

- [ ] **Step 4: Add failing renderer tests**

Render a two-frame 12-site motion with a known moving node. Assert:

- non-empty three-view and isometric PNGs;
- MP4 width 1246, height 720, FPS 30, two frames;
- first and second decoded frames differ;
- all four view panels contain non-background pixels.

- [ ] **Step 5: Implement OpenCV structure rendering**

Use fixed axis bounds over all frames. Draw 30 anti-aliased rod segments and
12 stable-size site circles in top XY, side XZ, front YZ, and isometric
projection. Use labels only in static images, not the animation. Refuse an
empty or partially written video.

- [ ] **Step 6: Package `qianji-package-vgt-motion`**

Arguments include QianJi site NPZ, robot, rig, desired control motion,
projected control motion, selected candidate JSON, output root, width, height,
and FPS.

- [ ] **Step 7: Run tests and commit**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest tests/test_vgt_sequence.py -q
uv run --python 3.12 --with '.[dev]' python -m pytest -q
git add pyproject.toml src/qianji_animal_motion/vgt_sequence.py \
  src/qianji_animal_motion/vgt_render.py \
  src/qianji_animal_motion/vgt_sequence_cli.py \
  tests/test_vgt_sequence.py tests/test_package.py
git commit -m "feat: package and render VGT model sequences"
```

---

### Task 6: Reachability Comparison and Final Acceptance Verifier

**Files:**
- Create: `experiments/39point_vgt_cat/compare_candidates.py`
- Create: `experiments/39point_vgt_cat/verify_case.py`
- Create: `tests/test_39point_experiment.py`

**Interfaces:**

```python
def compare_candidates(candidate_roots: list[Path]) -> dict: ...
def select_candidate(report: dict) -> dict: ...
def verify_case(case_root: Path) -> dict: ...
```

- [ ] **Step 1: Write failing candidate selection tests**

Use literal summaries to prove:

- an unreachable candidate is ineligible;
- error `0.050001` is ineligible;
- among eligible candidates scale `0.15` beats `0.10`;
- equal scale prefers higher feasible fraction, then lower maximum error;
- contraction, rig, and morphology parameters survive into the aggregate
  report.

- [ ] **Step 2: Implement aggregate reachability report**

Load each QianJi `run_summary.json`, `reachability_report.json`, robot, rig,
target motion, projected motion, and site NPZ. Validate hashes and dimensions.
Emit candidate table, selection rationale, contraction comparison, rig
comparison, morphology before/after comparison, and explicit eligibility
failures.

- [ ] **Step 3: Write failing full-case verifier tests**

Create a complete small synthetic case and mutate one requirement at a time:
missing role, NaN NPZ, 29 rods, mismatched hashes, desired/projected path
aliasing, wrong video FPS, wrong video frame count, missing limitation, and
partial output. Each mutation must make `passed: false` with a named check.

- [ ] **Step 4: Implement verifier**

The verifier checks all 12 requested deliverables and every acceptance gate.
It writes `reports/final_acceptance_report.json` with `passed`, timestamp,
source hashes, output hashes, individual checks, exact commands, and no
ambiguous skipped status.

- [ ] **Step 5: Run tests and commit**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest \
  tests/test_39point_experiment.py -q
uv run --python 3.12 --with '.[dev]' python -m pytest -q
git add experiments/39point_vgt_cat/compare_candidates.py \
  experiments/39point_vgt_cat/verify_case.py \
  tests/test_39point_experiment.py
git commit -m "feat: verify complete 39-point VGT cases"
```

---

### Task 7: One-Command Real Cat Experiment

**Files:**
- Create: `experiments/39point_vgt_cat/run_experiment.sh`
- Create: `experiments/39point_vgt_cat/README.md`
- Generated, ignored:
  `outputs/experiments/39point_vgt_cat_v1/**`

**Interfaces:**
- Consumes `ANIMAL_DATA_ROOT`, `QIANJI_ROOT`, and a new `OUTPUT_ROOT`.
- Produces the complete versioned case directory from the design.

- [ ] **Step 1: Write shell preflight and exact source paths**

Require the real video, H5, GLB, corrected six trajectory, local installed
commands, QianJi generator, reachability checker, and morphology optimizer.
Reject an existing output root before creating any directory.

- [ ] **Step 2: Export real observation and initial model**

Run `qianji-export-39-keypoints`; generate the abstract 12-site/30-rod QianJi
robot from the real GLB; generate bbox rig; validate exact counts.

- [ ] **Step 3: Generate scale-specific 39-point and control artifacts**

Create isolated lift/control directories for scales `0.05`, `0.10`, and
`0.15`. Prepare bbox/motion-informed rigs and zero/10-percent contraction
robots without overwriting any source.

- [ ] **Step 4: Run the bounded candidate matrix**

Run:

```text
scale_005_bbox_base_c000
scale_010_bbox_base_c000
scale_015_bbox_base_c000
scale_010_bbox_base_c010
scale_010_motion_base_c010
scale_010_motion_morph_c010
```

The morphology candidate uses QianJi
`controller/optimize_morphology_for_motion.py`; all others use
`controller/check_keypoint_reachability.py`.

- [ ] **Step 5: Compare, select, and publish final artifacts**

Generate aggregate `reports/reachability_report.json`; copy the selected
scale's `neutral_landmarks_39.json` and `keypoint_motion_3d_39.json`; publish
selected control map, desired target, projected target, robot, rig, and VGT
sequence through package commands rather than ambiguous manual renaming.

- [ ] **Step 6: Render and run acceptance verifier**

Produce 30 FPS 272-frame previews and
`reports/final_acceptance_report.json`. The script exits nonzero unless
`passed` is true.

- [ ] **Step 7: Execute the real experiment**

```bash
ANIMAL_DATA_ROOT="/absolute/main/animal-motion" \
QIANJI_ROOT="/absolute/QianJi" \
OUTPUT_ROOT="$PWD/outputs/experiments/39point_vgt_cat_v1" \
bash experiments/39point_vgt_cat/run_experiment.sh
```

- [ ] **Step 8: Inspect visual and quantitative evidence**

View the 2D preview, VGT three-view PNG, isometric PNG, and representative
frames from the MP4. Confirm no blank panel, incoherent rod, role loss, or
frame mismatch. Read the aggregate and final acceptance reports.

- [ ] **Step 9: Commit reproducibility files**

```bash
git add experiments/39point_vgt_cat/run_experiment.sh \
  experiments/39point_vgt_cat/README.md
git commit -m "exp: add reproducible 39-point cat VGT pipeline"
```

---

### Task 8: Documentation and Final Audit

**Files:**
- Modify: `README.md`
- Modify: `docs/data_contract.md`
- Create: `docs/39point_vgt_motion_pipeline.md`
- Modify:
  `docs/superpowers/plans/2026-07-30-39point-vgt-motion-pipeline.md`

- [ ] **Step 1: Document the exact pipeline and scientific boundary**

Explain Hunyuan3D mesh provenance, 39-point observation, neutral landmarks,
2.5D equation, observation/control separation, VGT sequence, desired versus
projected targets, and why no depth/dynamics claim is made.

- [ ] **Step 2: Document every schema and command**

List all required fields for the input manifest, 2D trajectory, neutral
landmarks, 39-point motion, control map, VGT NPZ/manifest, reachability report,
and acceptance report. Include the one-command real-case invocation.

- [ ] **Step 3: Run fresh final verification**

```bash
uv run --python 3.12 --with '.[dev]' python -m pytest -q
uv run --python 3.12 --with '.[dev]' qianji-export-39-keypoints --help
uv run --python 3.12 --with '.[dev]' qianji-lift-39-keypoints --help
uv run --python 3.12 --with '.[dev]' qianji-prepare-vgt-control --help
uv run --python 3.12 --with '.[dev]' qianji-package-vgt-motion --help
python experiments/39point_vgt_cat/verify_case.py \
  outputs/experiments/39point_vgt_cat_v1
git diff --check
git status --short --branch
```

- [ ] **Step 4: Audit every explicit acceptance requirement**

Record authoritative values in this plan: test count; 2D/3D frame and role
counts; NPZ shape; sites/rods; desired/projected distinct hashes; video FPS,
frames, dimensions and size; reachability status/error; NaN scan; output
hashes; branch and clean status.

- [ ] **Step 5: Commit documentation and verification record**

```bash
git add README.md docs/data_contract.md docs/39point_vgt_motion_pipeline.md \
  docs/superpowers/plans/2026-07-30-39point-vgt-motion-pipeline.md
git commit -m "docs: complete 39-point VGT pipeline record"
```

- [ ] **Step 6: Confirm clean final state**

```bash
git status --short --branch
git log --oneline --decorate -12
```

Only after every check is evidenced may the active goal be marked complete.
