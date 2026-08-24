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

The selected profile uses 12 IK iterations, 25.0 wrist translation weight, 3.0
per-axis wrist orientation weight, a 7.5 rad/s velocity cap, and three offline
smoothing passes. Relative to the pre-tuning profile, the full-suite evaluation
reduced worst-motion wrist position p95 by 29.0%, wrist orientation p95 by
68.5%, jitter RMS by 35.9%, maximum collision penetration by 16.9%, and total
colliding frames by 27.7%. Near-joint-limit samples increased from 8.38% to
8.58%.

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
