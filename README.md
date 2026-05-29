# Humanoid PPO Demo

This project runs the Gymnasium MuJoCo `Humanoid-v5` environment and includes a small PPO actor-critic trainer.

It works on macOS with the Python MuJoCo package. It does not require Isaac Sim or Isaac Lab.

## Files

- `run_humanoid.py` runs `Humanoid-v5` with random actions.
- `train_humanoid_ppo.py` trains or plays a PPO policy.
- `checkpoints/` stores saved policy checkpoints.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install "gymnasium[mujoco]" torch
```

If the local `.venv` already exists, you can use it directly without activating:

```bash
.venv/bin/python run_humanoid.py
```

## Run The Random Demo

```bash
.venv/bin/python run_humanoid.py --steps 200
```

Open the MuJoCo viewer:

```bash
.venv/bin/python run_humanoid.py --render
```

The random policy usually falls quickly. Increase `--steps` only sets the maximum episode length; it does not prevent termination after the humanoid falls.

## Train PPO

Quick smoke test:

```bash
.venv/bin/python train_humanoid_ppo.py --updates 5 --steps-per-update 512 --device cpu
```

Longer training run:

```bash
.venv/bin/python train_humanoid_ppo.py --updates 500 --steps-per-update 2048
```

By default, checkpoints are saved to:

```text
checkpoints/humanoid_ppo.pt
```

Humanoid is a difficult continuous-control task. Expect weak behavior at first; useful policies may require hundreds of thousands to millions of environment steps.

## Play A Checkpoint

Run without rendering:

```bash
.venv/bin/python train_humanoid_ppo.py --mode play --checkpoint checkpoints/humanoid_ppo.pt --play-steps 1000
```

Run with the MuJoCo viewer:

```bash
.venv/bin/python train_humanoid_ppo.py --mode play --checkpoint checkpoints/humanoid_ppo.pt --render
```

## Common Options

```bash
--updates 500              Number of PPO updates
--steps-per-update 2048    Environment steps collected before each update
--epochs 10                PPO optimization passes per rollout
--batch-size 256           Minibatch size
--hidden 256               Hidden layer size for actor and critic
--learning-rate 3e-4       Adam learning rate
--device auto              Uses MPS on Apple Silicon if available, otherwise CPU
--checkpoint PATH          Checkpoint path to save or load
```

## Notes

- `Humanoid-v5` has a 348-dimensional observation space and a 17-dimensional continuous action space.
- Training on CPU is slow. On Apple Silicon, `--device auto` will try PyTorch MPS.
- If rendering crashes in an editor terminal, try running the same command from the normal macOS Terminal.
