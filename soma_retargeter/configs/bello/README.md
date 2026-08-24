# Bello configuration and tuning

The external `bello_full_body_viewer.xml` MJCF defines the model contract used
by the retargeter. The configuration does not depend on GMR's solver or its
retargeted motions.

The robot description and meshes are intentionally not copied into this
repository. They are proprietary and require separate authorization. Supply an
authorized MJCF through `robot_model_path` or `BELLO_MJCF_PATH`.

The scaler was seeded from the SOMA-to-G1 calibration and Bello's neutral link
frames, then checked in the SOMA neutral pose. In particular, the foot scale is
the neutral root-to-ankle geometry ratio (0.82), rather than a transferred GMR
value. The Newton weights and offline
solver settings were selected independently by ablation over all ten bundled
SOMA BVHs (6,408 source frames). The evaluator measures source-task tracking,
joint-limit proximity, joint jitter and MuJoCo self-contact. Run the final
confirmation grid with:

```bash
uv sync --extra evaluation
BELLO_MJCF_PATH=/path/to/bello_full_body_viewer.xml \
  uv run python scripts/ablate_bello.py --preset confirmation
```

The selected profile uses 12 IK iterations, position-only elbow and hand
tracking, a smooth-joint objective weight of 5.0, a 7.5 rad/s velocity cap, and
three offline smoothing passes. The branch-selection stage uses elbow/hand
position weights of 20.0/15.0; it does not use SOMA limb-frame rotations, which
are not reliable indicators of Bello's limited arm twist. A dedicated two-bone
leg pass then restores the scaled foot targets without changing the whole-body
IK branch. On the AMASS squat
regression, median sole pitch fell from 13.9/12.2 degrees to 4.6/4.2 degrees,
matching the SOMA-X targets. On the bundled ten-motion suite it reduced
worst-case foot position p95 from 0.435 m to 0.247 m, jitter RMS from 1.022 to
0.984 degrees, and the colliding-frame fraction from 14.33% to 12.45%.

On the native-rate C10 AMASS regression, temporal regularization reduced right
elbow-pitch limit saturation from 58.9% to 0%; none of the four elbow degrees of
freedom remained pinned to a hard limit. Across the bundled SOMA suite, the
configuration reduced worst-case wrist-position p95 by 33.0%, near-limit
samples by 59.0%, and maximum collision penetration by 27.7%. Jitter increased
by 4.1% and the aggregate colliding-frame fraction by 13.3%, while the worst
motion's colliding-frame fraction fell by 2.1%. Wrist orientation is deliberately
not an objective because Bello cannot reproduce the full SOMA hand frame.

The MJCF home keyframe places the sole boxes about 57 mm above zero. The viewer
grounds that display pose from the configured sole geometry. Offline motions
are translated by a single reference ground offset, with per-frame penetration
clamping, so stance frames touch zero without flattening jump trajectories. On
the bundled suite, median lowest-sole height is 0 mm for every clip and the
high-jump clip retains about 0.64 m of clearance.

These are comparative kinematic metrics, not claims of dynamic feasibility.
Standing pickups still contain substantial self-contact and should be validated
in the downstream controller before hardware playback.

The output contract is a floating root followed by 25 logical hinge positions.
Ankle pitch and roll remain logical coordinates; differential motor mixing is a
downstream control concern.
