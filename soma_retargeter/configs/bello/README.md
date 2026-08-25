# Bello configuration and tuning

The external `bello_full_body_viewer.xml` MJCF defines the model contract used
by the retargeter. The robot description and meshes are not redistributed;
supply an authorized MJCF with `robot_model_path` or `BELLO_MJCF_PATH`.

Bello uses the same vanilla Newton retargeting path as G1: one `ik_map`, one IK
solve per frame, the standard smooth-joint and joint-limit objectives, followed
by the standard two-bone foot stabilizer and joint-limit clamp. Bello-specific
code is limited to model loading and output formatting. There are no staged IK,
axis-weighted objectives, offline passes, velocity caps, or motion-grounding
passes.

The scaler was initialized from Bello's neutral link geometry and checked in
the SOMA neutral pose. Its root scale is lowered from the floating MJCF home
height using a ten-clip support-height sweep; the selected value places the
median low-stance sole height at the ground plane without a motion-grounding
pass. The single-stage task weights start from the upstream G1 pattern and were
ablated on all ten bundled SOMA BVHs (6,408 source frames).
The pelvis orientation anchors the body frame, while thigh and shin tasks are
position-led to avoid equivalent but mechanically flipped leg solutions. Foot
orientation remains active to preserve sole attitude. The pelvis orientation
weight is deliberately below the translation weight: a stronger orientation
anchor selected poor whole-body branches on several AMASS clips. Increasing
the foot orientation weight beyond its selected value caused the same failure.
The selected arm map uses weak upper-arm and hand orientation tasks, while
forearm orientation is zero-weighted. Bello's elbow frame cannot reproduce the
full SOMA forearm frame; asking the whole-body solver to match it caused wrong
IK branches and severe upstream twist.

Against the initial vanilla Bello map on all 6,408 bundled BVH frames, the final
map reduced self-collision frames from 4.57% to 2.43%, maximum penetration from
48.6 mm to 20.0 mm, near-limit samples from 4.59% to 3.98%, worst hand position
p95 from 455 mm to 379 mm, worst hand orientation p95 from 63.1 to 60.4 degrees,
worst jitter RMS from 1.84 to 1.65 degrees, and velocity p99 from 810 to 705
degrees/second. The worst foot-position p95 increased from 54 mm to 70 mm on
the extreme body stretch clip, an accepted tradeoff for the branch, collision,
and smoothness gains.

Five four-second AMASS checks (run, squat, punch-boxing, crawl, and roundhouse)
reduced mean self-collision frames from 50.0% to 36.3%, mean near-limit samples
from 18.3% to 14.1%, and worst joint-acceleration p99 by 76%. The ACCAD run's
worst hand-position p95 fell from 258 mm to 13 mm. These are comparative
kinematic checks, not claims of dynamic feasibility.

Reproduce the bounded tuning grid with:

```bash
uv sync --extra evaluation
BELLO_MJCF_PATH=/path/to/bello_full_body_viewer.xml \
  uv run python scripts/ablate_bello.py --preset broad
```

Use `--preset confirmation --max-frames 0` to evaluate the final candidates on
every frame. The default 240-frame cap keeps exploratory runs memory-bounded.

The output contract is a floating root followed by 25 logical hinge positions.
Ankle pitch and roll remain logical coordinates; differential motor mixing is a
downstream control concern.
