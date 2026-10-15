# RoboFlow-VLA

A lightweight Vision-Language-Action (VLA) policy that fuses visual observations, language instructions, and robot states to predict continuous robot actions using a diffusion-based action head.

## Architecture

The model consists of four core components:

- **Encoders** (`models/encoders.py`): Encode images, text instructions, and robot states.
- **Feature Fusion** (`models/fusion.py`): Fuse multimodal representations using an MLP.
- **Diffusion Action Head** (`models/diffusion_head.py`): Generate continuous action predictions.
- **VLA Policy** (`models/vla_diffusion_policy.py`): Integrate the encoders, fusion module, and action head into a unified policy.

## Pipeline

1. **Data Collection:** Collect expert demonstrations from Meta-World environments.
2. **Training:** Train the policy on demonstration trajectories.
3. **Evaluation:** Evaluate the trained policy in simulation and save rollout videos.

## Setup

Requires Python 3.10.

```bash
conda create -n roboflow-vla python=3.10
conda activate roboflow-vla

git clone https://github.com/tanvi2594/RoboFlow-VLA
cd RoboFlow-VLA

pip install -r requirements.txt
```

## Data Collection

Collect expert demonstrations for the Meta-World `push-v3` task:

```bash
python -m scripts.collect_data \
  --env-name push-v3 \
  --camera-name corner \
  --episodes 100 \
  --max-steps 100 \
  --output-path data/metaworld_push_bc.npz
```

## Training

Train the diffusion-based VLA policy:

```bash
python -m scripts.train \
  --dataset-path data/push_v3.npz \
  --epochs 50 \
  --batch-size 64 \
  --save-path checkpoints/model.pt \
  --device cpu
```

## Evaluation

Evaluate the trained policy in Meta-World simulation:

```bash
python -m scripts.test \
  --checkpoint checkpoints/model.pt \
  --env-name push-v3 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos
```

## Technical Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Policy Learning:** Diffusion-based action generation
- **Multimodal Inputs:** Images, language instructions, and robot states
- **Simulation:** Meta-World
- **Data Format:** NumPy (`.npz`)

## Core Research Areas

- Multimodal representation learning
- Vision-conditioned robotic manipulation
- Language-conditioned action prediction
- Imitation learning from expert demonstrations
- Robot policy evaluation in simulation