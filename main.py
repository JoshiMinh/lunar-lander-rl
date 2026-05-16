import os
import sys
import glob
import time
from pathlib import Path

import numpy as np
import torch

from src.game.environment import VastSpaceLander
from src.train.agent import DQNAgent
from src.train.trainer import train_all_variants, train_variant


def scan_models_directory(models_dir='models'):
    """Scan models directory for available .pth files and their scores."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    pth_files = sorted(glob.glob(os.path.join(models_dir, '*.pth')))
    models_with_scores = []

    for pth_file in pth_files:
        filename = os.path.basename(pth_file)
        avg_score = None

        base_name = filename.replace('.pth', '')
        if base_name.endswith('_model'):
            scores_name = base_name[:-6] + '_scores.npy'
        else:
            scores_name = base_name + '_scores.npy'

        scores_file = os.path.join(models_dir, scores_name)
        if os.path.exists(scores_file):
            try:
                scores = np.load(scores_file)
                avg_score = float(np.mean(scores))
            except Exception as exc:
                print(f"Warning: Could not load scores from {scores_file}: {exc}")

        models_with_scores.append((filename, avg_score))

    return models_with_scores


def parse_model_config_from_name(filename):
    """Parse DQN variant configuration from model filename."""
    filename_lower = filename.lower()
    config = {
        'double_dqn': True,
        'dueling': True,
    }

    if 'vanilla' in filename_lower:
        config['double_dqn'] = False
        config['dueling'] = False
    elif 'dueling' in filename_lower and 'double' not in filename_lower:
        config['double_dqn'] = False
        config['dueling'] = True
    elif 'double' in filename_lower and 'dueling' not in filename_lower:
        config['double_dqn'] = True
        config['dueling'] = False
    elif 'd3qn' in filename_lower or ('double' in filename_lower and 'dueling' in filename_lower):
        config['double_dqn'] = True
        config['dueling'] = True

    return config


def build_interactive_menu():
    """Build interactive CLI menu for model selection and configuration."""
    models = scan_models_directory('models')

    if not models:
        print("❌ No trained models found in 'models/' directory.")
        print("   Please train a model first using: python main.py")
        return None

    print("\n" + "=" * 70)
    print("🚀 LUNAR LANDER DEMO - Model Selection")
    print("=" * 70)
    print("\n📦 Available Models:\n")

    for idx, (filename, avg_score) in enumerate(models, 1):
        score_str = f"  (avg score: {avg_score:.2f})" if avg_score is not None else "  (scores not available)"
        print(f"  {idx}. {filename}{score_str}")

    print()

    while True:
        try:
            choice = input(f"Select model (1-{len(models)}): ").strip()
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx][0]
                model_path = os.path.join('models', selected_model)
                break
            print(f"❌ Please enter a number between 1 and {len(models)}")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number between 1 and {len(models)}")

    print(f"\n✅ Selected: {selected_model}")

    suggested_config = parse_model_config_from_name(selected_model)

    print("\n🔧 DQN Configuration:\n")
    default_double = "Y" if suggested_config['double_dqn'] else "N"
    while True:
        choice = input(f"Use Double DQN? (Y/N) [default: {default_double}]: ").strip().upper()
        if choice == "":
            double_dqn = suggested_config['double_dqn']
            break
        if choice in ["Y", "N"]:
            double_dqn = (choice == "Y")
            break
        print("❌ Please enter Y or N")

    default_dueling = "Y" if suggested_config['dueling'] else "N"
    while True:
        choice = input(f"Use Dueling DQN? (Y/N) [default: {default_dueling}]: ").strip().upper()
        if choice == "":
            dueling = suggested_config['dueling']
            break
        if choice in ["Y", "N"]:
            dueling = (choice == "Y")
            break
        print("❌ Please enter Y or N")

    print("\n⏱️  Episode Configuration:\n")
    episode_options = [5, 10, 20, 100]
    for idx, ep_count in enumerate(episode_options, 1):
        print(f"  {idx}. {ep_count} episodes")

    while True:
        try:
            choice = input(f"\nSelect episodes (1-{len(episode_options)}) [default: 2 (10 episodes)]: ").strip()
            if choice == "":
                episodes = 10
                break
            ep_idx = int(choice) - 1
            if 0 <= ep_idx < len(episode_options):
                episodes = episode_options[ep_idx]
                break
            print(f"❌ Please enter a number between 1 and {len(episode_options)}")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number between 1 and {len(episode_options)}")

    print("\n" + "=" * 70)
    print("📋 Configuration Summary:")
    print("=" * 70)
    print(f"  Model:        {selected_model}")
    print(f"  Double DQN:   {'✓' if double_dqn else '✗'}")
    print(f"  Dueling DQN:  {'✓' if dueling else '✗'}")
    print(f"  Episodes:     {episodes}")
    print("=" * 70 + "\n")

    return {
        'model_path': model_path,
        'double_dqn': double_dqn,
        'dueling': dueling,
        'episodes': episodes,
    }


def load_state_dict_flexible(model, path, device=None):
    """Load a checkpoint into model, copying only parameters with matching shapes."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    target_dict = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in checkpoint.items():
        if key in target_dict:
            if value.shape == target_dict[key].shape:
                target_dict[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append((key, value.shape, target_dict[key].shape))
        else:
            skipped_keys.append((key, None, None))

    model.load_state_dict(target_dict)
    return loaded_keys, skipped_keys


def run_demo(config):
    """Run a demo episode batch for the selected checkpoint."""
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
        dueling=config['dueling'],
    )

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
    except Exception as exc:
        print(f"❌ Error loading model: {exc}")
        env.close()
        return

    episode_count = config.get('episodes', 10)
    print(f"\n🎮 Starting demo with {episode_count} episodes...\n")

    episode_rewards = []
    success_count = 0

    for i in range(episode_count):
        state, _ = env.reset()
        score = 0
        for _ in range(env.max_episode_steps):
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
                print(f"Episode {i + 1}/{episode_count} | Status: {color}{status.upper()}{reset} | Reward: {score:.2f}")
                if status == 'success':
                    success_count += 1
                episode_rewards.append(score)
                break

    env.close()

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


def build_quick_run_config():
    """Return the default one-click demo configuration."""
    best_model_path = os.path.join('models', 'd3qn_model_best.pth')
    selected_model_path = best_model_path if os.path.exists(best_model_path) else os.path.join('models', 'd3qn_model.pth')
    return {
        'model_path': selected_model_path,
        'double_dqn': True,
        'dueling': True,
        'episodes': 10,
    }


def run_quick_demo():
    """Start the default D3QN demo without extra prompts."""
    print("\n⚡ Quick Run selected: D3QN demo with default settings.\n")
    config = build_quick_run_config()
    if not os.path.exists(config['model_path']):
        print(f"❌ Model file not found: {config['model_path']}")
        return
    run_demo(config)


def run_train_interactive():
    """Prompt for training options and dispatch to the selected training routine."""
    try:
        eps = input("Number of episodes [default: 3000]: ").strip()
        episodes = int(eps) if eps else 3000
    except ValueError:
        episodes = 3000

    print("\nSelect training mode:")
    print("  1) D3QN")
    print("  2) Double DQN")
    print("  3) Dueling DQN")
    print("  4) Train all 3 models")
    mode = input("Choose (1-4) [default: 1]: ").strip()

    reset_input = input("Reset training (delete existing checkpoint)? (y/N): ").strip().lower()
    reset_flag = (reset_input == 'y')

    print(f"Starting training: episodes={episodes}, reset={reset_flag}")
    if mode == "2":
        train_variant("double_dqn", n_episodes=episodes, reset=reset_flag)
    elif mode == "3":
        train_variant("dueling_dqn", n_episodes=episodes, reset=reset_flag)
    elif mode == "4":
        train_all_variants(n_episodes=episodes, reset=reset_flag)
    else:
        train_variant("d3qn", n_episodes=episodes, reset=reset_flag)


if __name__ == "__main__":
    try:
        # Top-level choice: Quick Run, Demo, or Train
        print("\nSelect action:\n  0) Quick Run (D3QN demo, one click)\n  1) Demo (run trained model)\n  2) Train (start training)")
        choice = input("Choose (0/1/2) [default: 0]: ").strip()
        if choice == "2":
            run_train_interactive()
        elif choice == "1":
            config = build_interactive_menu()
            if config is None:
                print("❌ No models available. Exiting.")
                sys.exit(1)
            if not os.path.exists(config['model_path']):
                print(f"❌ Model file not found: {config['model_path']}")
                sys.exit(1)
            run_demo(config)
        elif choice == "0" or choice == "":
            run_quick_demo()
        else:
            print("❌ Please enter 0, 1, or 2.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Exiting gracefully...")
