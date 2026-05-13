import os
import sys
import time
import torch
import numpy as np
from src.game import VastSpaceLander
from src.agent import DQNAgent
from src.utils import load_state_dict_flexible


def run_demo(config):
    """Run demo given a config dict (delegated from main)."""
    env = VastSpaceLander(render_mode='human')
    env.max_episode_steps = 5000
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        seed=0,
        device=device,
        double_dqn=config['double_dqn'],
        dueling=config['dueling']
    )

    # Load weights flexibly
    try:
        loaded_keys, skipped = load_state_dict_flexible(agent.qnetwork_local, config['model_path'], device=device)
        if loaded_keys:
            print(f"✅ Loaded {len(loaded_keys)} parameters from checkpoint.")
        if skipped:
            print(f"⚠ Skipped {len(skipped)} parameters due to shape/key mismatch.")
            for item in skipped[:6]:
                if isinstance(item, tuple) and item[1] is not None:
                    print(f"   - {item[0]}: checkpoint shape={item[1]} target_shape={item[2]}")
    except FileNotFoundError:
        print(f"❌ Model file not found: {config['model_path']}")
        env.close()
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        return

    episode_count = config.get('episodes', 10)
    print(f"\n🎮 Starting demo with {episode_count} episodes...\n")

    episode_rewards = []
    success_count = 0

    for i in range(episode_count):
        state, _ = env.reset()
        score = 0
        for t in range(env.max_episode_steps):
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            time.sleep(0.01)

            if getattr(env, 'user_quit', False):
                print("\n[Q] Pressed - Force Shutting Down...")
                env.close()
                return

            if terminated or truncated:
                status = info.get('mission_status', 'failed')
                color = "\033[92m" if status == 'success' else "\033[91m"
                reset = "\033[0m"
                print(f"Episode {i+1}/{episode_count} | Status: {color}{status.upper()}{reset} | Reward: {score:.2f}")
                if status == 'success':
                    success_count += 1
                episode_rewards.append(score)
                break

    env.close()

    # Summary (guard against empty list)
    if episode_rewards:
        avg = np.mean(episode_rewards)
        mx = np.max(episode_rewards)
        mn = np.min(episode_rewards)
    else:
        avg = mx = mn = 0.0

    print("\n" + "=" * 70)
    print("📊 Demo Summary")
    print("=" * 70)
    print(f"  Total Episodes:    {episode_count}")
    print(f"  Successful:        {success_count}/{episode_count}")
    print(f"  Average Reward:    {avg:.2f}")
    print(f"  Max Reward:        {mx:.2f}")
    print(f"  Min Reward:        {mn:.2f}")
    print("=" * 70 + "\n")


def run_train_interactive():
    """Prompt minimal train options and invoke training function from src.train."""
    try:
        from src.train import train
    except Exception as e:
        print(f"❌ Training module not available: {e}")
        return

    try:
        eps = input("Number of episodes [default: 3000]: ").strip()
        episodes = int(eps) if eps else 3000
    except ValueError:
        episodes = 3000

    reset_input = input("Reset training (delete existing checkpoint)? (y/N): ").strip().lower()
    reset_flag = (reset_input == 'y')

    print(f"Starting training: episodes={episodes}, reset={reset_flag}")
    train(n_episodes=episodes, reset=reset_flag)
