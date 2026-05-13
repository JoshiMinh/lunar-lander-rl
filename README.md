# 🚀 LunarLanderRL: Deep Reinforcement Learning for Precision Descent

> **Last Updated:** May 13, 2026
>
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-20B2AA?logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a custom **VastSpaceLander** reinforcement-learning environment, a **Dueling Double DQN** agent, and an archived LunarLander-v3 report that has now been merged into this README so the notebook can be removed safely.

---

## 🌌 The VastSpaceLander Environment

The main codebase simulates a larger, harsher landing scenario than the stock Gymnasium LunarLander.

### State Space
The agent observes a 9-dimensional normalized vector:
1. Horizontal position relative to the landing pad.
2. Vertical position relative to the landing pad.
3. Horizontal velocity.
4. Vertical velocity.
5. Lander angle.
6. Angular velocity.
7. Left leg contact flag.
8. Right leg contact flag.
9. Remaining fuel level.

### Environment Characteristics
- High gravity for faster, more difficult descents.
- Fuel-limited thrusters that can force a ballistic fall.
- Procedurally generated terrain with a narrow landing zone.
- Long episode horizon for careful landing control.
- Visual landing guidance tuned for the custom renderer.

### Reward Structure
- Successful landing bonus.
- Fuel-saving bonus.
- Speed bonus for landing sooner.
- Descent shaping reward when centered over the pad.
- Small living cost to avoid hovering.
- Crash penalty for failed landings.

---

## 🧠 Model Architecture

The agent uses a **Dueling Double Deep Q-Network (D3QN)** design:

- Dueling streams separate state value $V(s)$ from action advantage $A(s,a)$.
- Double DQN selects actions with the local network and evaluates them with the target network.
- Experience replay stores transitions in a $10^5$-entry buffer.
- Soft target updates track the online network with $\tau = 0.001$.

```mermaid
graph LR
    S[9D State Input] --> H1[Dense 128 + ReLU]
    H1 --> H2[Dense 128 + ReLU]
    H2 --> V[Value Stream V(s)]
    H2 --> A[Advantage Stream A(s,a)]
    V --> Q[Q(s,a) = V + (A - mean(A))]
    A --> Q
```

### Reward Shaping Highlights
- Base landing reward.
- Fuel conservation bonus.
- Faster-landing bonus.
- Vertical descent shaping over the pad.
- Small per-step penalty to discourage hovering.
- Failure penalty for crashes.

---

## 🛠️ Getting Started

### Installation
```bash
pip install gymnasium[box2d] torch matplotlib tqdm pandas pygame
```

### Training
The training script supports resume-from-checkpoint behavior and now stores logs in `models/`.

```bash
python train.py --episodes 3000
python train.py --reset
```

### Demo
The demo now features an **interactive model selection menu** that lets you:
1. Choose any trained model from the `models/` directory
2. Select DQN variant configuration (Double DQN, Dueling DQN, or both)
3. Choose the number of episodes to run (5, 10, 20, or 100)

Simply run:
```bash
python main.py
```

The menu will display all available models with their average training scores (if available), then guide you through the configuration options. Example output:

```
======================================================================
🚀 LUNAR LANDER DEMO - Model Selection
======================================================================

📦 Available Models:

  1. checkpoint.pth                 (scores not available)
  2. d3qn_model.pth                 (avg score: 184.80)
  3. double_dqn_model.pth           (avg score: 177.69)
  4. dueling_dqn_model.pth          (avg score: 173.47)
  5. vanilla_dqn_model.pth          (avg score: 187.14)

Select model (1-5): 
```

### Artifact Layout
All saved artifacts now live in `models/`:
- `models/checkpoint.pth`
- `models/training_log.csv`
- `models/*_model.pth`
- `models/*_scores.npy`

---

## 📘 Archived Notebook Report

The old `DQN_LunarLander_v3.ipynb` content is preserved here as a report-style summary.

### 1. Problem Setup
The notebook studied the standard **LunarLander-v3** environment as a reinforcement-learning control problem.

- State space: 8 continuous dimensions.
- Action space: 4 discrete thrust actions.
- Success criterion: average reward of at least 200 over 100 episodes.

### 2. Theory
The report covered the core RL and DQN equations:

$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s, a_t = a\right]$$

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

$$\mathcal{L}(\theta) = \mathbb{E}\left[(y - Q(s, a; \theta))^2\right]$$

It also explained:
- Experience replay.
- Target networks.
- Double DQN.
- Dueling DQN.
- D3QN as the combination of Double + Dueling.

### 3. Implementation Notes
The notebook implemented the usual DQN training stack:
- A 2-layer Q-network with ReLU activations.
- A replay buffer for transition sampling.
- A DQN agent with epsilon-greedy exploration.
- Soft target updates.
- Gradient clipping for stability.

### 4. Training Configuration
The notebook used these representative hyperparameters:
- Buffer size: $10^5$
- Batch size: 64
- Discount factor: 0.99
- Soft update factor: $10^{-3}$
- Learning rate: $5 \times 10^{-4}$
- Episodes: 2000
- Max steps per episode: 1000
- Epsilon schedule: start 1.0, end 0.01, decay 0.995

### 5. Experiments
Four variants were compared under the same seed and hyperparameters:
- Vanilla DQN.
- Double DQN.
- Dueling DQN.
- D3QN.

The notebook recorded learning curves, summary tables, and per-variant score files.

### 6. Evaluation
The report evaluated the best model with greedy inference and visualized:
- Reward distribution.
- Success rate.
- A live demo rollout.

### 7. Conclusion
The notebook concluded that Double DQN and Dueling DQN both improve over Vanilla DQN, while D3QN is typically the strongest and most stable of the four.

---

## 🤖 CI/CD Automated Training

The GitHub Actions workflow trains the agent on pushes to `main` and can resume from `models/checkpoint.pth` plus `models/training_log.csv`.

- Auto-resume keeps long training runs moving across commits.
- Manual reset is available from the Actions tab when training should restart from scratch.

---

## 📝 License
This project is licensed under the MIT License. It is intended for reinforcement-learning research and demonstration.
