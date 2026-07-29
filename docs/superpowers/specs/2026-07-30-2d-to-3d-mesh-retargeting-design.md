# Experimental 2D-to-3D Mesh Retargeting Design

## Goal

Bridge `qianji-animal-motion` six-point image trajectories into QianJi's
`qianji-keypoint-trajectory-v1` contract by transferring body-relative motion
onto the neutral six-point rig of a VGT morphology generated from the same
animal mesh.

This is an experimental retargeting path, not monocular metric 3D
reconstruction. It must never label an unobserved camera-depth coordinate as a
measured value.

## Evidence And Scope

The workspace report separates the system into:

1. animal image/video/mesh sources;
2. 2D, 2.5D, and 3D morphology-bank candidates;
3. a canonical `robot.json` plus semantic keypoint mapping;
4. keypoint control and QianJi/MuJoCo validation.

QianJi already provides the downstream pieces:

- `morph_generator/generate.py 3d-mesh` converts a mesh to `robot.json`;
- `controller/check_keypoint_reachability.py --auto-rig quadruped_bbox`
  produces `rig_keypoints.json`;
- QianJi consumes `qianji-keypoint-trajectory-v1` frames containing
  `[x, y, z, confidence]`;
- reachability, morphology optimization, and MuJoCo control consume that
  contract.

The missing piece is therefore an adapter from
`qianji.keypoint_trajectory_2d` to the QianJi motion contract. This branch adds
only that adapter and its evidence. It does not duplicate QianJi's mesh-to-VGT
generator or controller.

The linked ChatGPT conversation could not be opened through the available
browser because the custom `chatgpt-conversation://` scheme was blocked. The
design therefore uses the workspace report image, both repositories, their
data contracts, and current generated artifacts as authoritative evidence.

## Considered Approaches

### Direct monocular depth prediction

Run a learned 2D-to-3D pose model and treat its output as world coordinates.
This is rejected for the first experiment because the input has only six
points, no camera calibration, no matching animal skeleton, and no verified
scale. It would create a precise-looking but weakly supported result.

### Body-relative 2.5D retargeting onto a mesh-derived rig

Use the 2D trajectory only for observable longitudinal and vertical
articulation. Preserve the mesh-derived rig's lateral/depth coordinate. Remove
image translation, in-plane rotation, and camera zoom by expressing each frame
in a spine-aligned local frame. This is the selected approach because it is
deterministic, auditable, immediately consumable by QianJi, and honest about
the missing depth.

### Six-anchor mesh-cage deformation

Use the six points as a deformation cage and animate every mesh vertex. This is
useful for visualization but not yet useful to QianJi's rod controller, and
six anchors are too sparse for credible leg deformation. It remains a later
visual experiment after the canonical keypoint adapter is validated.

## Inputs

The adapter consumes:

- a `qianji.keypoint_trajectory_2d` JSON trajectory;
- a QianJi `robot.json` generated from the target animal mesh;
- a `rig_keypoints.json` mapping the six semantic names to distinct robot
  sites;
- an optional reference frame, defaulting to the trajectory's manually
  confirmed identity-anchor frame;
- a dimensionless motion scale, default `0.25`.

The six required roles are:

- `spine_rear`
- `spine_front`
- `front_left_foot`
- `front_right_foot`
- `rear_left_foot`
- `rear_right_foot`

## Coordinate Transfer

For every frame with valid spine points:

1. define the 2D origin as the midpoint between `spine_rear` and
   `spine_front`;
2. define the local longitudinal unit vector from rear to front;
3. define the local vertical vector perpendicular to the longitudinal vector,
   choosing the sign that points toward image-up;
4. express each keypoint in this local basis and divide by that frame's torso
   length in pixels;
5. subtract the corresponding reference-frame local coordinate.

The QianJi neutral frame is derived from the rig:

1. the neutral forward vector is the normalized vector from the rear-spine
   site to the front-spine site;
2. the neutral up vector starts as global `+Z` and is orthogonalized against
   the forward vector;
3. the neutral lateral vector is retained only as provenance; no image-derived
   displacement is applied along it;
4. the neutral torso length supplies metric scale.

For role `k`, the output is:

```text
p3_k(t) = neutral_k
        + motion_scale * torso_length_3d
        * (delta_longitudinal_k(t) * forward_3d
           + delta_vertical_k(t) * up_3d)
```

This transfers articulation while removing camera translation, in-plane
rotation, and uniform zoom. It preserves the mesh/VGT neutral width and never
estimates camera depth.

## Invalid Data Policy

QianJi's current consumers do not use the confidence field to suppress a
target. Omitting a point can also cause neighboring-frame interpolation.
Therefore an invalid 2D point is emitted at its neutral 3D site with
`confidence = 0.0`. The lift report records every substitution.

If a frame cannot define a spine frame, the entire frame is emitted at the
neutral pose with zero confidence. No temporal interpolation or last-value
carry is introduced by the adapter.

## Outputs

The CLI writes a staged, atomically published directory containing:

- `keypoint_motion.json`: QianJi `qianji-keypoint-trajectory-v1`;
- `mesh_binding.json`: source hashes, mesh-derived robot/rig association,
  neutral site coordinates, and the 3D basis;
- `lift_report.json`: assumptions, reference frame, scale, invalid
  substitutions, motion ranges, and explicit depth observability status.

The report must state:

```text
reconstruction_kind: body_relative_2_5d_retarget
metric_depth_observed: false
camera_calibrated: false
global_translation_preserved: false
```

## CLI

```bash
qianji-lift-keypoints \
  --trajectory path/to/keypoint_trajectory_2d_corrected.json \
  --robot-json path/to/robot.json \
  --rig path/to/rig_keypoints.json \
  --output path/to/lifted_motion \
  --motion-scale 0.25
```

`--reference-frame` accepts an integer. When omitted, the tool uses
`identity_anchor.frame_idx`; if that is absent it selects the valid frame whose
torso length is closest to the valid median.

The command refuses to overwrite any owned output. Source files are hashed
before processing and checked again before publication.

## Validation Experiment

The experiment uses the current corrected cat trajectory, the converted cat
GLB, and QianJi's existing mesh pipeline:

1. generate an `abstract` VGT from the GLB;
2. generate a quadruped bbox rig;
3. lift the 272-frame 30 FPS trajectory at motion scales `0.10`, `0.25`, and
   `0.50`;
4. render QianJi keypoint previews;
5. run QianJi reachability on each scale;
6. compare feasible/marginal/unreachable counts, maximum keypoint error, edge
   violation, and clipping;
7. retain the most useful scale as the experimental example, without claiming
   biological 3D accuracy.

## Test Strategy

Unit tests use synthetic trajectories and neutral rigs to prove:

- unchanged 2D pose maps exactly to the neutral 3D pose;
- image translation, rotation, and uniform scale are removed;
- an observed foot lift moves in neutral-frame `+up`;
- no lateral displacement is inferred;
- reference-frame selection is deterministic;
- invalid points and invalid spine frames become neutral zero-confidence
  targets and are reported;
- missing/duplicate rig sites and malformed coordinates are rejected;
- the CLI publishes all three files atomically and refuses overwrite.

The full existing test suite must remain green. The real experiment is a
separate integration check against QianJi and is recorded with commands,
hashes, metrics, and generated previews.
