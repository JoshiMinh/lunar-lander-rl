# Lunar Lander RL

Deep reinforcement-learning experiments for a custom Lunar Lander style environment.

The project now uses a compact source layout:

`src/__init__.py` is the only package initializer.
All gameplay code lives under `src/game`.
All model definitions live under `src/models`.
All training, replay-buffer, and CLI code lives under `src/train`.

## What It Does

The repository trains and runs a DQN-based agent against the custom `VastSpaceLander` environment.
It supports vanilla, Double DQN, Dueling DQN, and D3QN variants.
Saved checkpoints and score files are stored in `models/`.

## Layout

- `main.py` starts the interactive demo or training flow.
- `src/game/environment.py` defines the environment.
- `src/game/terrain.py` and `src/game/renderer.py` support the simulation.
- `src/models/double_dqn.py` defines the standard Q-network.
- `src/models/dueling_dqn.py` defines the dueling network.
- `src/models/d3qn.py` combines the dueling architecture with Double DQN training.
- `src/train/agent.py` contains the DQN agent logic.
- `src/train/memory.py` contains the replay buffer.
- `src/train/trainer.py` contains the training loops.
- `src/train/cli.py` contains the interactive menu and demo launcher.

## Install

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

If you prefer a manual install, the main runtime packages are `gymnasium[box2d]`, `torch`, `numpy`, `pandas`, `pygame`, and `tqdm`.

## Run

Start the interactive entrypoint:

```bash
python main.py
```

The main menu has two choices:

1. Run: pick a trained model and launch the game with it.
2. Train: choose one model variant or train all variants, then set the episode count.

Demo mode lets you select one checkpoint from `models/` and then choose the run settings.
Training mode lets you pick `D3QN`, `Double DQN`, `Dueling DQN`, or train all three in sequence.
The final prompt in training asks for the episode count.

## Artifacts

Training writes checkpoints and logs into `models/`.

- `models/training_log.csv`
- `models/*_model.pth`
- `models/*_scores.npy`
- `models/*_training_log.csv`

Typical files already present in the folder are:

- `d3qn_model.pth`
- `double_dqn_model.pth`
- `dueling_dqn_model.pth`
- `d3qn_scores.npy`
- `double_dqn_scores.npy`
- `dueling_dqn_scores.npy`
- `training_log.csv`

## Notes

The repository is intended for reinforcement-learning experimentation and small-scale agent comparisons.
The custom environment is more aggressive than stock LunarLander, so training can take time.
The demo renderer is configured for human observation when a compatible display is available.

## License

This project is distributed under the MIT License.
