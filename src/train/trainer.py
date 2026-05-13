import os
if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import time

from src.game.environment import VastSpaceLander
from src.train.agent import DQNAgent

VARIANT_CONFIGS = {
    'd3qn': {
        'double_dqn': True,
        'dueling': True,
        'save_path': 'models/d3qn_model.pth',
        'log_path': 'models/d3qn_training_log.csv',
    },
    'double_dqn': {
        'double_dqn': True,
        'dueling': False,
        'save_path': 'models/double_dqn_model.pth',
        'log_path': 'models/double_dqn_training_log.csv',
    },
    'dueling_dqn': {
        'double_dqn': False,
        'dueling': True,
        'save_path': 'models/dueling_dqn_model.pth',
        'log_path': 'models/dueling_dqn_training_log.csv',
    },
}


def train(n_episodes=3000, max_t=5000, eps_start=1.0, eps_end=0.05, eps_decay=0.995, save_path='models/d3qn_model.pth', log_path='models/training_log.csv', reset=False, max_time=None, double_dqn=True, dueling=True):
    """Cloud-Optimized Deep Q-Learning with CSV logging, resume support, and headless mode."""
    env = VastSpaceLander()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, device=device, double_dqn=double_dqn, dueling=dueling)
    
    start_episode = 1
    rewards = []
    rewards_window = deque(maxlen=100)
    eps = eps_start
    history = []

    if reset:
        print("🔄 Reset flag detected. Starting training from scratch...")
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"   ✓ Deleted old model file: {save_path}")
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f"   ✓ Deleted old logs: {log_path}")
    elif os.path.exists(save_path):
        print(f"📦 Model found at {save_path}. Attempting to resume...")
        try:
            agent.qnetwork_local.load_state_dict(torch.load(save_path, map_location=device))
            agent.qnetwork_target.load_state_dict(torch.load(save_path, map_location=device))
            
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                if not log_df.empty:
                    last_episode = int(log_df.iloc[-1]['episode'])
                    start_episode = last_episode + 1
                    n_episodes = last_episode + n_episodes
                    eps = float(log_df.iloc[-1]['epsilon'])
                    history = log_df.to_dict('records')
                    print(f"   ✓ Resumed from Episode {start_episode}. Target: {n_episodes} (Epsilon: {eps:.4f})")
                else:
                    print(f"   ⚠ Model loaded, but log file empty. Starting from Episode 1 with existing weights.")
            else:
                print(f"   ⚠ Model loaded, but no log file. Starting from Episode 1 with existing weights.")
        except Exception as e:
            print(f"   ✗ Failed to load model: {e}. Starting from scratch.")
    else:
        print("🆕 No model file found. Starting fresh training from Episode 1.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    interrupted = False
    timed_out = False
    total_start_time = time.time()
    
    pbar = tqdm(range(start_episode, n_episodes + 1), desc="Training")
    try:
        for i_episode in pbar:
            state, _ = env.reset()
            score = 0
            start_time = time.time()
            
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            
            duration = time.time() - start_time
            rewards_window.append(score)
            rewards.append(score)
            eps = max(eps_end, eps_decay * eps)
            
            history.append({
                'episode': i_episode,
                'reward': score,
                'avg_reward': np.mean(rewards_window),
                'epsilon': eps,
                'duration': duration
            })

            pbar.set_postfix({
                'AvgReward': f'{np.mean(rewards_window):.1f}',
                'Eps': f'{eps:.2f}'
            })
            
            if i_episode % 50 == 0:
                torch.save(agent.qnetwork_local.state_dict(), save_path)
                pd.DataFrame(history).to_csv(log_path, index=False)
                
            if len(rewards_window) >= 100 and np.mean(rewards_window) >= 200.0:
                print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Reward: {np.mean(rewards_window):.2f}')
                torch.save(agent.qnetwork_local.state_dict(), save_path)
                pd.DataFrame(history).to_csv(log_path, index=False)
                break
            
            if max_time and (time.time() - total_start_time > max_time):
                print(f"\nReached maximum training time ({max_time}s). Saving progress and exiting...")
                timed_out = True
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user (Ctrl+C). Saving progress...")
    finally:
        if history:
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            pd.DataFrame(history).to_csv(log_path, index=False)
            if interrupted or timed_out:
                last_ep = history[-1]['episode']
                print(f"Saved model and logs up to episode {last_ep}.")
        elif interrupted:
            print("No completed episodes yet; nothing to save.")
            
    return history


def train_variant(variant_name, n_episodes=3000, max_t=5000, eps_start=1.0, eps_end=0.05, eps_decay=0.995, reset=False, max_time=None):
    """Train one named model variant and save it into models/."""
    if variant_name not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant_name}")

    cfg = VARIANT_CONFIGS[variant_name]
    print(f"\n=== Training {variant_name.upper()} ===")
    return train(
        n_episodes=n_episodes,
        max_t=max_t,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        save_path=cfg['save_path'],
        log_path=cfg['log_path'],
        reset=reset,
        max_time=max_time,
        double_dqn=cfg['double_dqn'],
        dueling=cfg['dueling'],
    )


def train_all_variants(n_episodes=3000, max_t=5000, eps_start=1.0, eps_end=0.05, eps_decay=0.995, reset=False, max_time=None):
    """Train all three requested model variants one after another."""
    results = {}
    for variant_name in ('d3qn', 'double_dqn', 'dueling_dqn'):
        results[variant_name] = train_variant(
            variant_name,
            n_episodes=n_episodes,
            max_t=max_t,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            reset=reset,
            max_time=max_time,
        )
    return results


__all__ = ["train", "train_variant", "train_all_variants", "VARIANT_CONFIGS", "DQNAgent"]
