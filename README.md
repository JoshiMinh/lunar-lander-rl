# Lunar Lander RL ??

A reinforcement learning playground for a custom Lunar Lander-style environment built around DQN variants.

## Overview ??

This repository trains and evaluates agents against the `VastSpaceLander` environment.
It includes support for vanilla DQN-style training, Double DQN, Dueling DQN, and D3QN.
The code is organized so the game environment, model definitions, and training utilities live in separate modules.

## What You Can Do ???

- Launch the interactive demo menu from `main.py`.
- Select a trained checkpoint from `models/`.
- Run the lander in human-rendered demo mode.
- Train one model variant or train all variants in sequence.
- Resume training from a saved checkpoint when a matching model file exists.
- Log rewards and episode timing to CSV for later analysis.

## Project Layout ??

- `main.py` provides the interactive menu, demo launcher, and model selection flow.
- `src/game/environment.py` defines the custom environment.
- `src/game/constants.py` stores environment constants.
- `src/game/terrain.py` builds terrain and landing zones.
- `src/game/renderer.py` handles rendering and visual feedback.
- `src/models/d3qn.py` defines the combined Double DQN + Dueling architecture.
- `src/models/double_dqn.py` defines the Double DQN network.
- `src/models/dueling_dqn.py` defines the Dueling DQN network.
- `src/train/agent.py` contains the agent policy and learning logic.
- `src/train/memory.py` implements replay memory.
- `src/train/trainer.py` contains the training loops and variant helpers.
- `models/` stores checkpoints, score arrays, and training logs.

## Requirements ??

Install the Python dependencies from `requirements.txt`.
The main runtime packages are `gymnasium[box2d]`, `torch`, `numpy`, `pandas`, `pygame`, and `tqdm`.
A working Box2D-capable environment is required for the custom lander simulation.

## Setup ??

Create and activate a virtual environment, then install the dependencies.

```bash
pip install -r requirements.txt
```

If you are setting up a fresh local environment, install from the project root so the package imports resolve correctly.

## Run the App ??

Start the interactive application with:

```bash
python main.py
```

From there, you can choose one of two paths:

- Demo: pick a saved model and run it in the renderer.
- Train: choose a DQN variant and start a training session.

## Training Modes ??

The trainer currently exposes three named variants:

- `d3qn` for Double DQN plus Dueling heads.
- `double_dqn` for Double DQN without dueling.
- `dueling_dqn` for Dueling DQN without the Double DQN target update path.

You can also train all supported variants in sequence from the interactive flow.
Training supports checkpoint resume behavior when an output file already exists and `reset` is not enabled.

## Programmatic Training ??

If you want to call the trainer directly, import it from `src.train.trainer`.

```python
from src.train.trainer import train_variant

train_variant('d3qn', n_episodes=3000, max_time=21600, reset=False)
```

The helper accepts the same options used by the interactive flow, including episode count, max time, and reset behavior.

## Outputs ??

Training writes artifacts into `models/`.

- `*_model.pth` files store checkpoints.
- `*_scores.npy` files store score history for evaluation.
- `*_training_log.csv` files store episode logs.
- `training_log.csv` is the default consolidated log produced by training runs.

Existing sample artifacts in the repository can be used immediately for demo playback or further training.

## Behavior Notes ???

The environment uses a headless-safe configuration for training by default.
Rendering for demo mode uses the human display path when available.
The custom terrain is more demanding than the stock LunarLander task, so useful policies may take time to emerge.

## Tips ??

- Use a smaller episode count when smoke-testing a code change.
- Keep `reset=False` if you want to continue from an existing checkpoint.
- Use `reset=True` when you want to discard prior progress and start fresh.
- Watch the CSV logs if you want a quick view of reward trends over time.

## Troubleshooting ??

If imports fail, confirm that you are running commands from the repository root.
If Box2D or rendering fails on your machine, reinstall dependencies inside a clean virtual environment.
If demo mode cannot find a checkpoint, verify that the desired `.pth` file is present in `models/`.

## Contributing ??

Keep changes focused and avoid committing local virtual environments or cache directories.
When adding new training outputs, prefer filenames that match the existing `models/` conventions so the menu can discover them automatically.
