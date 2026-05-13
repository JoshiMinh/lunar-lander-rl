from src.runner import run_demo, run_train_interactive
from src.utils import build_interactive_menu


if __name__ == "__main__":
    try:
        # Top-level choice: Demo or Train
        print("\nSelect action:\n  1) Demo (run trained model)\n  2) Train (start training)")
        choice = input("Choose (1/2) [default: 1]: ").strip()
        if choice == "2":
            run_train_interactive()
        else:
            # Run demo flow
            config = build_interactive_menu()
            if config is None:
                print("❌ No models available. Exiting.")
                sys.exit(1)
            if not os.path.exists(config['model_path']):
                print(f"❌ Model file not found: {config['model_path']}")
                sys.exit(1)
            run_demo(config)

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Exiting gracefully...")
