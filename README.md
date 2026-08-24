# SOMA Retargeter
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SOMA Retargeter Banner](assets/docs/banner.gif)

Convert [SOMA](https://github.com/NVlabs/SOMA-X) human motion captures into humanoid robot joint animation. Takes BVH motion files as input and produces robot-playable CSV joint data as output using GPU-optimized inverse kinematics via [Newton](https://github.com/newton-physics/newton) and high-performance computation with [NVIDIA Warp](https://github.com/NVIDIA/warp).

The retargeting pipeline handles proportional human-to-robot scaling, multi-objective IK solving with joint limits, feet stabilization to maintain ground contact, and per-DOF joint limit clamping. It currently supports SOMA as the input skeleton and Unitree G1 (29 DOF) or Bello (25 DOF) as output robots.

SOMA Retargeter is part of the [SOMA body model](https://github.com/NVlabs/SOMA-X) ecosystem for humanoid motion data.

> **Note:** This project is in active development. The API may change between releases as the design is refined.

## Requirements

- **Python:** 3.12
- **Git LFS:** Installed and initialized for asset downloads
- **OS:** Windows (x86-64) and Linux (x86-64, aarch64)
- **GPU:** NVIDIA GPU (Maxwell or newer), driver 545+ (CUDA 12). No local CUDA Toolkit installation required.

## Installation

<details>

<summary>Setup instructions</summary>

### Method 1 (conda + pip)

#### 1. Create and Activate Conda Environment

```bash
conda create -n soma-retargeter python=3.12 -y
conda activate soma-retargeter
```

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Install the Library

```bash
pip install .
```

### Method 2 (uv)

#### 1. Install uv

Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if `uv` is not yet installed.

#### 2. Download LFS Assets

```bash
git lfs pull
```

#### 3. Sync the Project

`uv sync` creates an isolated `.venv` virtual environment inside the project directory, installs the correct Python version and resolves all dependencies.

```bash
uv sync
```

### Method 3 (Nix flake)

Enter the development shell:

```bash
nix develop
```

The flake supplies Python 3.12, `uv`, Git LFS, FFmpeg, build tools, Tk, and the
Linux graphics libraries needed by the viewer. It automatically syncs `.venv`
from `uv.lock`, so `uv run` commands work immediately. Run `git lfs pull` once
if the motion assets were not cloned.

The optional AMASS shell installs the pinned SOMA-X conversion stack without
adding its PyTorch dependency to the normal development environment:

```bash
nix develop .#amass
```

With `direnv` and `nix-direnv` installed, approve the checked-in `.envrc` once:

```bash
direnv allow
```

The same flake and locked uv environment will then activate whenever you enter the repository.

### Platform-specific notes

**Note (Linux):** For the GUI viewer to work, install `tkinter`

```bash
sudo apt-get install python3.12-tk
```

**Note (Windows):** If `imgui-bundle` fails to install, the Microsoft Visual C++ Redistributables may be missing. Download from the [official Microsoft documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

</details>

## Motion Data

This repo includes 10 sample BVH/CSV pairs in `assets/motions/` for immediate testing.

For large-scale motion data, see the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) published by [Bones Studio](https://huggingface.co/bones-studio). SEED provides a large-scale collection of human motions on the SOMA uniform-proportion skeleton, which is the expected input format for this tool. The G1 robot motion data included in SEED was retargeted using SOMA Retargeter.

## Quick Start

> When using **uv** (Method 2), replace `python` with `uv run` in the commands below.

### Interactive viewer (OpenGL)

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer gl
```

![Interactive viewer interface](assets/docs/interactive-viewer-screenshot.png)

The viewer displays the source SOMA motion alongside the retargeted robot in a 3D viewport. Use the right panel to load BVH files, run retargeting, and save CSV output. Playback controls at the bottom allow scrubbing, speed adjustment, and looping. Toggle visibility of the skinned mesh, skeleton, joint axes, and positioning gizmos.

### Batch conversion (headless)

Process a folder of BVH files without a display. Set `import_folder` and `export_folder` in the config file, then run:

```bash
python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer null
```

Batch mode recursively finds all `.bvh` files in the import folder, processes them in configurable batch sizes, and writes CSV files to the export folder mirroring the input directory structure.

## Bello target

Bello descriptions and meshes are not redistributed by this project. Point the retargeter at an authorized `bello_full_body_viewer.xml` MJCF generated by the Bello asset pipeline. The retargeting weights are tuned independently on this repository's bundled SOMA motion suite.

### Interactive Bello visualizer

```bash
export BELLO_MJCF_PATH="$HOME/bello_mujoco/deps/GMR/assets/bello/mjcf/bello_full_body_viewer.xml"
uv run python ./app/bvh_to_csv_converter.py \
  --config ./assets/bello_bvh_to_csv_converter_config.json \
  --viewer gl
```

The viewer starts in Bello's MJCF `home` pose. Use **BVH Motion → Load → Retarget** to display the SOMA source and Bello result together. The timeline supports play/pause, scrubbing, reverse/variable speed, and looping, and the result can be saved as a named Bello CSV from the same panel.

To capture that same Newton `ViewerGL` presentation headlessly, render the
canonical CSV directly. This preserves the interactive viewer's mesh shading,
checker floor, contact shadows, and actor-relative side camera; it does not
translate the motion through another playback engine.

```bash
export BELLO_MJCF_PATH="$HOME/bello_mujoco/deps/GMR/assets/bello/mjcf/bello_full_body_viewer.xml"
uv run soma-render-bello \
  ./outputs/amass-bello/squat03_stageii.csv \
  ./outputs/amass-bello/squat03_stageii.mp4
```

The command expects the canonical 120 Hz Bello CSV by default and renders at
30 fps. Use `--source-fps` for an explicitly resampled CSV, and camera or output
options from `uv run soma-render-bello --help` when needed.

### Batch conversion

```bash
export BELLO_MJCF_PATH="$HOME/bello_mujoco/deps/GMR/assets/bello/mjcf/bello_full_body_viewer.xml"
uv run python ./app/bvh_to_csv_converter.py \
  --config ./assets/bello_bvh_to_csv_converter_config.json \
  --viewer null
```

The Bello configuration writes the normal named CSV. The optional
`export_gmr_pickle` compatibility output is disabled by default to keep exports
small; when enabled, it adds a legacy pickle containing `fps`, `root_pos`,
`root_rot` (xyzw), and 25 logical `dof_pos` coordinates. Differential ankle
motor mixing remains a downstream control concern; the retargeter exports
logical ankle pitch and roll.

### AMASS SMPL-X input

AMASS support uses the official `py-soma-x==0.2.1` topology and pose inversion
pipeline to produce the same 78-joint SOMA representation consumed by the
Newton retargeter. Enter `nix develop .#amass` or run `uv sync --extra amass`,
then point the converter at the separately licensed SMPL-X body models:

```bash
export BELLO_MJCF_PATH="$HOME/bello_mujoco/deps/GMR/assets/bello/mjcf/bello_full_body_viewer.xml"
uv run --extra amass python ./app/amass_to_csv_converter.py \
  "$HOME/amass/KIT/572/squat03_stageii.npz" \
  --body-model-dir "$HOME/amass/body_models" \
  --output-dir ./outputs/amass-bello
```

SOMA-X assets are downloaded to the Hugging Face cache on first use. Pass
`--soma-assets` to use an existing asset checkout. `--max-seconds` bounds
validation runs, while `--export-gmr-pickle` writes the optional visualization
compatibility format in addition to the named Bello CSV. Native AMASS frame
rates are preserved by default; `--target-fps` is an explicit resampling
override. Identity-specific SMPL-X proportions are normalized to the uniform
SOMA rig expected by the retargeting parameters unless
`--no-normalize-stature` is passed.

The AMASS profile also reconstructs upper-arm and palm frames from SOMA-X
anatomical landmarks, removing inverse-pose limb-roll ambiguity before robot
IK. During detected foot contact it preserves sole heading while leveling sole
roll/pitch in a bounded robot-space pass; swing-foot orientation is untouched.
Use `--no-stabilize-anatomical-frames` or `--no-level-contact-feet` for
ablations, and `--hand-orientation-weight` to override wrist tracking strength.

To reproduce the Bello parameter ablation, install the evaluation extra and run
the confirmation preset:

```bash
uv sync --extra evaluation
BELLO_MJCF_PATH=/path/to/bello_full_body_viewer.xml \
  uv run python scripts/ablate_bello.py --preset confirmation
```

## Code Overview

### `app/`

| File | Description |
|------|-------------|
| `bvh_to_csv_converter.py` | Main entry point. Drives both interactive and headless batch retargeting modes. |
| `amass_to_csv_converter.py` | AMASS SMPL-X to SOMA-X to robot batch conversion. |

### `soma_retargeter/`

| Module | Description |
|--------|-------------|
| `animation/` | Core data structures for skeletons, animation buffers, IK, and skinned meshes. |
| `assets/` | File I/O for BVH, CSV, and USD formats. |
| `pipelines/` | Retargeting pipeline: IK solving, feet stabilization, and joint limit clamping. |
| `robotics/` | Human-to-robot scaling and robot output formatting. |
| `renderers/` | Visualization for the interactive viewer. |
| `utils/` | Math, pose, coordinate conversion, Newton and Warp helpers. |
| `configs/` | JSON configuration for retargeting, scaling, and feet stabilization parameters. |

## Related Work

SOMA Retargeter is a support tool within the SOMA ecosystem for humanoid motion data:

* [SOMA Body Model](https://github.com/NVlabs/SOMA-X) - Parametric human body model with standardized skeleton, mesh, and shape parameters
* [GEM-X](https://github.com/NVlabs/GEM-X) - Human motion estimation from video
* [Kimodo](https://github.com/nv-tlabs/kimodo) - Kinematic motion diffusion model for text and constraint-driven 3D human and robot motion generation
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots
* [SONIC](https://nvlabs.github.io/GEAR-SONIC/) - Whole-body control for humanoid robots, training locomotion and interaction policies

## Acknowledgments

This project draws inspiration and builds upon excellent open-source work, including:
* [GMR](https://github.com/YanjieZe/GMR) - General Motion Retargeting
* [PyRoki](https://pyroki-toolkit.github.io/) - A Modular Toolkit for Robot Kinematic Optimization

## License

This codebase is licensed under [Apache-2.0](LICENSE).

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
