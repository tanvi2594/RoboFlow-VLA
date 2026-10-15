# RoboFlow-VLA

A lightweight Vision-Language-Action (VLA) policy that combines visual observations, language instructions, and robot states to predict continuous robot actions using a diffusion-based action head.

## Architecture

The model consists of four core components:

- **Encoders** (`models/encoders.py`): Encode images, text instructions, and robot states.
- **Feature Fusion** (`models/fusion.py`): Fuse multimodal representations using an MLP.
- **Diffusion Action Head** (`models/diffusion_head.py`): Generate continuous action predictions.
- **VLA Policy** (`models/vla_diffusion_policy.py`): Integrate the encoders, fusion module, and action head into a unified policy.

## Pipeline

1. **Data Collection:** Collect expert demonstrations from Meta-World environments.
2. **Training:** Train the policy on demonstration trajectories.
3. **Evaluation:** Evaluate the trained policy through simulated rollouts.
4. **Analysis:** Measure task success, inference latency, and robustness under visual perturbations.

## Setup

**Requirements:** Python 3.10, Conda, and the dependencies listed in `requirements.txt`.

```bash
conda create -n roboflow-vla python=3.10
conda activate roboflow-vla

git clone https://github.com/tanvi2594/RoboFlow-VLA.git
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
  --dataset-path data/metaworld_push_bc.npz \
  --epochs 50 \
  --batch-size 64 \
  --save-path checkpoints/model.pt \
  --device cpu
```

> **Note:** The dataset path must match the file produced during data collection. Confirm that the training script accepts these arguments and that the collected dataset matches the expected format.

## Evaluation

Evaluate the trained policy in Meta-World simulation and save rollout videos:

```bash
python -m scripts.test \
  --checkpoint checkpoints/model.pt \
  --env-name push-v3 \
  --episodes 100 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos
```

## Results and Evaluation

The policy is designed to be evaluated on the Meta-World `push-v3` object manipulation task using visual observations, language instructions, and robot states.

### Target Performance Benchmarks

The following values are **experimental targets, not measured results**.

| Metric | Target |
|---|---:|
| Task success rate | ≥ 80% |
| Mean episode reward | ≥ 75% of achievable environment reward |
| Evaluation episodes | 100 |
| Mean inference latency | ≤ 100 ms per action |
| Success rate under visual perturbations | ≥ 65% |
| Success-rate drop under visual perturbations | ≤ 15 percentage points |

### Evaluation Protocol

- **Task:** Meta-World `push-v3`
- **Evaluation episodes:** 100 held-out rollouts
- **Maximum episode length:** 150 steps
- **Inputs:** Camera observations, language instructions, and robot states
- **Metrics:** Task success rate, mean episode reward, inference latency, and robustness to visual perturbations
- **Artifacts:** Rollout videos and per-episode evaluation logs

### Planned Ablation Studies

1. **Multimodal conditioning:** Compare the complete policy with variants that remove language or robot-state inputs.
2. **Demonstration quality:** Compare policies trained on the original demonstrations and a curated subset.
3. **Visual robustness:** Test controlled changes in lighting, image noise, and camera observations.
4. **Inference efficiency:** Measure per-action latency after model warm-up and document the hardware used.

Actual results should be reported only after completing the experiments. Comparisons should use consistent evaluation settings and held-out episodes.

## Technical Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Policy Learning:** Diffusion-based action generation
- **Multimodal Inputs:** Images, language instructions, and robot states
- **Simulation:** Meta-World
- **Data Processing:** NumPy (`.npz`)

## Core Research Areas

- Multimodal representation learning
- Vision-conditioned robotic manipulation
- Language-conditioned action prediction
- Imitation learning from expert demonstrations
- Robustness evaluation for robotic policies
- Demonstration data quality and policy evaluation
