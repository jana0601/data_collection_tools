"""
Gesture Classification Inference Launcher.
Easy way to run GUI or CLI inference applications.
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def check_dependencies(mode: str) -> bool:
    """Verify required Python packages are installed."""
    try:
        import sklearn  # noqa: F401
    except Exception as e:
        print("[ERROR] Missing dependency: scikit-learn")
        print("Install it with one of:")
        print("  pip install -r ml/requirements.txt")
        print("  pip install scikit-learn")
        print(f"Details: {e}")
        return False

    if mode == 'gui':
        try:
            import PySide6  # noqa: F401
        except Exception as e:
            print("[ERROR] Missing dependency for GUI: PySide6")
            print("Install it with:")
            print("  pip install PySide6")
            print(f"Details: {e}")
            return False

    return True


def check_models():
    """Check if trained models exist."""
    # Resolve models directory relative to this script so it works from any CWD
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models_2class"
    
    # Accept both multiclass and 2-class metadata filenames
    metadata_candidates = [
        models_dir / "gesture_multiclass_metadata.json",
        models_dir / "gesture_2class_metadata.json",
    ]
    
    metadata_path = next((p for p in metadata_candidates if p.exists()), None)
    
    if metadata_path is None:
        print("[ERROR] Trained models not found!")
        print("Checked:")
        for p in metadata_candidates:
            print(f"  - {p}")
        print("Please run the training script first:")
        print("  python train_multiclass_mediapipe.py")
        return False
    
    print("[OK] Trained models found")
    return True


def run_gui():
    """Run GUI inference application."""
    try:
        from inference_app import main as gui_main
        print("[STARTING] Starting GUI inference application...")
        gui_main()
    except ImportError as e:
        print(f"[ERROR] Failed to import GUI dependencies: {e}")
        print("Make sure PySide6 is installed:")
        print("  pip install PySide6")


def run_cli():
    """Run CLI inference application."""
    try:
        from inference_cli import main as cli_main
        print("[STARTING] Starting CLI inference application...")
        cli_main()
    except ImportError as e:
        print(f"[ERROR] Failed to import CLI dependencies: {e}")


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='Gesture Classification Inference Launcher')
    parser.add_argument('--mode', choices=['gui', 'cli'], default='gui',
                       help='Inference mode: gui (default) or cli')
    parser.add_argument('--check-models', action='store_true',
                       help='Only check if models exist')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GESTURE CLASSIFICATION INFERENCE LAUNCHER")
    print("=" * 60)
    
    # Check Python dependencies first
    if not check_dependencies(args.mode):
        return

    # Check models
    if not check_models():
        return
    
    if args.check_models:
        print("[OK] Models check completed")
        return
    
    # Run inference application
    if args.mode == 'gui':
        run_gui()
    else:
        run_cli()


if __name__ == "__main__":
    main()
