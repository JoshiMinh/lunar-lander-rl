"""Utility functions for model management and interactive menu."""

import os
import glob
import numpy as np
from pathlib import Path
import torch


def scan_models_directory(models_dir='models'):
    """Scan models directory for available .pth files and their scores.
    
    Returns:
        list: List of tuples (filename, avg_score or None)
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    pth_files = sorted(glob.glob(os.path.join(models_dir, '*.pth')))
    models_with_scores = []
    
    for pth_file in pth_files:
        filename = os.path.basename(pth_file)
        avg_score = None
        
        # Try to load corresponding .npy scores file
        # Handle naming convention: {name}_model.pth -> {name}_scores.npy
        #                           {name}.pth -> {name}_scores.npy
        base_name = filename.replace('.pth', '')
        if base_name.endswith('_model'):
            # Remove _model suffix for score file lookup
            scores_name = base_name[:-6] + '_scores.npy'  # Remove "_model" and add "_scores.npy"
        else:
            scores_name = base_name + '_scores.npy'
        
        scores_file = os.path.join(models_dir, scores_name)
        if os.path.exists(scores_file):
            try:
                scores = np.load(scores_file)
                avg_score = float(np.mean(scores))
            except Exception as e:
                print(f"Warning: Could not load scores from {scores_file}: {e}")
        
        models_with_scores.append((filename, avg_score))
    
    return models_with_scores


def parse_model_config_from_name(filename):
    """Parse DQN variant configuration from model filename.
    
    Args:
        filename (str): Model filename (e.g., 'double_dqn_model.pth')
    
    Returns:
        dict: Configuration with 'double_dqn' and 'dueling' keys
    """
    filename_lower = filename.lower()
    
    # Initialize with defaults
    config = {
        'double_dqn': True,
        'dueling': True
    }
    
    # Parse filename for variant hints
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
    """Build interactive CLI menu for model selection and configuration.
    
    Returns:
        dict: Configuration dict with keys:
            - 'model_path': full path to selected model
            - 'double_dqn': bool
            - 'dueling': bool
            - 'episodes': int
        or None if user cancels
    """
    # Scan for available models
    models = scan_models_directory('models')
    
    if not models:
        print("❌ No trained models found in 'models/' directory.")
        print("   Please train a model first using: python train.py")
        return None
    
    # Display available models
    print("\n" + "=" * 70)
    print("🚀 LUNAR LANDER DEMO - Model Selection")
    print("=" * 70)
    print("\n📦 Available Models:\n")
    
    for idx, (filename, avg_score) in enumerate(models, 1):
        score_str = f"  (avg score: {avg_score:.2f})" if avg_score is not None else "  (scores not available)"
        print(f"  {idx}. {filename}{score_str}")
    
    print()
    
    # Model selection
    while True:
        try:
            choice = input(f"Select model (1-{len(models)}): ").strip()
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx][0]
                model_path = os.path.join('models', selected_model)
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(models)}")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number between 1 and {len(models)}")
    
    print(f"\n✅ Selected: {selected_model}")
    
    # Get suggested config from filename
    suggested_config = parse_model_config_from_name(selected_model)
    
    # Double DQN selection
    print("\n🔧 DQN Configuration:\n")
    default_double = "Y" if suggested_config['double_dqn'] else "N"
    while True:
        choice = input(f"Use Double DQN? (Y/N) [default: {default_double}]: ").strip().upper()
        if choice == "":
            double_dqn = suggested_config['double_dqn']
            break
        elif choice in ["Y", "N"]:
            double_dqn = (choice == "Y")
            break
        else:
            print("❌ Please enter Y or N")
    
    # Dueling selection
    default_dueling = "Y" if suggested_config['dueling'] else "N"
    while True:
        choice = input(f"Use Dueling DQN? (Y/N) [default: {default_dueling}]: ").strip().upper()
        if choice == "":
            dueling = suggested_config['dueling']
            break
        elif choice in ["Y", "N"]:
            dueling = (choice == "Y")
            break
        else:
            print("❌ Please enter Y or N")
    
    # Episode count selection
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
            else:
                print(f"❌ Please enter a number between 1 and {len(episode_options)}")
        except ValueError:
            print(f"❌ Invalid input. Please enter a number between 1 and {len(episode_options)}")
    
    # Confirmation
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
        'episodes': episodes
    }


def load_state_dict_flexible(model, path, device=None):
    """Load a checkpoint into `model`, copying only parameters with matching shapes.

    Returns a tuple (loaded_keys, skipped_keys). Does not raise on shape mismatches.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Support checkpoints with 'state_dict' wrapper
        checkpoint = checkpoint['state_dict']

    target_dict = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for k, v in checkpoint.items():
        if k in target_dict:
            if v.shape == target_dict[k].shape:
                target_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append((k, v.shape, target_dict[k].shape))
        else:
            # key not present in target model
            skipped_keys.append((k, None, None))

    model.load_state_dict(target_dict)
    return loaded_keys, skipped_keys
