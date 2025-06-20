# Minimal Unitree G1 DRL Locomotion Training

This is a minimal Unitree G1 training environment designed for fast setup, including terrain curriculum, heightfield measurements, sim-to-sim, and deployment.

## Installation

Follow these steps to set up the environment and install the required packages.

### 1. Create Conda Environment

We recommend using Conda or Mamba to manage the environment.

```bash
conda create -n g1_gym python=3.8
conda activate g1_gym
```

### 2. Install Dependencies

First, install PyTorch with CUDA support. Please refer to the official [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command matching your CUDA version. For example, for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Then, install other dependencies:
```bash
pip install numpy matplotlib
```

### 2.2 Install Isaac Gym
Isaac Gym is a rigid body simulation and training framework provided by Nvidia.

#### 2.2.1 Download
Download **Isaac Gym** from Nvidia's official website.

#### 2.2.2 Install
After extracting the package, navigate to the `isaacgym/python` folder and install it using the following commands:
```bash
cd isaacgym/python
pip install -e .
```

#### 2.2.3 Verify Installation
Run the following command. If a window opens displaying 1080 balls falling, the installation was successful:
```bash
cd examples
python 1080_balls_of_solitude.py
```
If you encounter any issues, refer to the official documentation at `isaacgym/docs/index.html`.

### 3. Install Local Packages

The required robotics packages are included in this repository. Install them using pip in editable mode. The order of installation is important.

```bash
pip install -e isaacgym/python
pip install -e rsl_rl
pip install -e legged_gym
```

### 4. Test the Installation

You can test your installation by running one of the example scripts provided with IsaacGym:

```bash
cd isaacgym/python/examples
python 1080_balls_of_solitude.py
```
or
```bash
python joint_monkey.py
```

### Troubleshooting

#### `libpython` error

If you encounter an error related to `libpython`, you may need to set your `LD_LIBRARY_PATH`.

First, find your conda environment's library path:
```bash
conda info --envs
```
This will list your environments and their locations. Find the path for `g1_gym`.

Then, export the `LD_LIBRARY_PATH`, replacing `/path/to/conda/envs/g1_gym` with the actual path from the previous command:
```bash
export LD_LIBRARY_PATH=/path/to/conda/envs/hvgym/lib:$LD_LIBRARY_PATH
```
To make this change permanent, add this line to your `~/.bashrc` or `~/.zshrc` file. 

## User Guide

### 1. Training

Run the following command to start training:
```bash
python legged_gym/scripts/train.py --task=g1_full
```

#### Parameter Description
*   `--task`: Required parameter; values can be `g1`, `g1_full`, etc.
*   `--headless`: Set to `True` to run in headless mode for higher efficiency. Defaults to `False` (GUI mode).
*   `--resume`: Resume training from a checkpoint in the logs.
*   `--experiment_name`: Name of the experiment to run/load.
*   `--run_name`: Name of the run to execute/load.
*   `--load_run`: Name of the run to load; defaults to the latest run.
*   `--checkpoint`: Checkpoint number to load; defaults to the latest file.
*   `--num_envs`: Number of environments for parallel training.
*   `--seed`: Random seed.
*   `--max_iterations`: Maximum number of training iterations.
*   `--sim_device`: Simulation computation device (e.g., `cpu` or `cuda:0`).
*   `--rl_device`: Reinforcement learning computation device (e.g., `cpu` or `cuda:0`).

**Default Training Result Directory:** `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

To visualize the training results in the simulator, run the following command:
```bash
python legged_gym/scripts/play.py --task=g1_full
```

![Training Visualization](assets/vis_1.gif)

#### Description
*   The parameters for `play.py` are the same as for `train.py`.
*   By default, it loads the latest model from the most recent run in the experiment folder.
*   You can specify other models to load using the `--load_run` and `--checkpoint` flags. 

---

### 3. Sim-to-Sim Deployment (MuJoCo)

To deploy the trained policy in a MuJoCo simulation, run the following commands:

```bash
cd legged_gym/sim2mujoco/deploy_mujoco
python deploy_mujoco.py g1_27dof.yaml
```

![Sim2Sim Deployment](assets/sim2sim.gif)

---

## To-Do
- [x] Release training code
- [x] Release sim-to-sim deployment code
- [ ] Release robot deployment code