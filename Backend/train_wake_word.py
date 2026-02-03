#!/usr/bin/env python3
"""
Train a custom wake word model for "Hey Icarus" using OpenWakeWord.

This script automates the entire training pipeline:
1. Downloads required datasets
2. Sets up Piper TTS for synthetic audio generation
3. Generates positive and negative training samples
4. Augments samples with noise/reverb
5. Trains the wake word model
6. Exports to ONNX format

Usage:
    python train_wake_word.py --setup      # First time setup (download datasets)
    python train_wake_word.py --generate   # Generate synthetic clips
    python train_wake_word.py --augment    # Augment clips with noise/reverb
    python train_wake_word.py --train      # Train the model
    python train_wake_word.py --all        # Run entire pipeline

Output:
    models/hey_icarus/hey_icarus.onnx - Copy this to Client folder for detection
"""

import os
import sys
import subprocess
import logging
import shutil
import argparse
import urllib.request
import zipfile
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WakeWordTrainer")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
PIPER_DIR = BASE_DIR / "piper_sample_generator"
OUTPUT_DIR = MODELS_DIR / "hey_icarus"

# Training configuration for "Hey Icarus"
TRAINING_CONFIG = {
    "model_name": "hey_icarus",
    "target_phrase": ["hey icarus"],
    "n_samples": 5000,              # Synthetic training samples (5000 minimum, 20000+ recommended)
    "n_samples_val": 500,           # Validation samples
    "tts_batch_size": 32,           # TTS generation batch size
    "augmentation_batch_size": 16,  # Augmentation batch size
    "augmentation_rounds": 1,       # Rounds of data augmentation
    "model_type": "dnn",            # Model architecture: "dnn" or "rnn"
    
    # Custom negative phrases (phonetically similar - should NOT trigger)
    "custom_negative_phrases": [
        "hey iris",
        "hey siri",
        "hey google",
        "hey alexa",
        "hay icarus",
        "he icarus",
        "hey i curse",
        "hey ikarus",
        "high icarus",
        "hey marcus",
        "hey nicholas",
        "hey daedalus",
        "the icarus",
        "by icarus",
    ],
    
    # Model training parameters
    "max_steps": 50000,
    "max_negative_weight": 500,
    "target_false_positives_per_hour": 0.2,
}

# Dataset URLs from HuggingFace
DATASETS = {
    "validation_features": {
        "url": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
        "filename": "validation_set_features.npy",
        "size_mb": 500,
        "required": True,
    },
    "negative_features": {
        "url": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "filename": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "size_mb": 8000,
        "required": False,  # Optional but improves quality significantly
    },
}


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Hey Icarus - Wake Word Training Pipeline                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This script trains a custom wake word model using OpenWakeWord.             ║
║                                                                              ║
║  Pipeline Steps:                                                             ║
║    1. Setup    - Download datasets and Piper TTS                             ║
║    2. Generate - Create synthetic audio samples via TTS                      ║
║    3. Augment  - Add noise, reverb, pitch variations                         ║
║    4. Train    - Train the neural network model                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """Check if required packages are installed."""
    # Core training dependencies
    required = {
        "openwakeword": "openwakeword",
        "torch": "torch",
        "torchaudio": "torchaudio",
        "yaml": "pyyaml",
        "numpy": "numpy",
        "soundfile": "soundfile",
        "datasets": "datasets",
    }
    
    # Piper sample generator dependencies
    piper_deps = {
        "piper": "piper-tts",
        "webrtcvad": "webrtcvad-wheels",
        "audiomentations": "audiomentations",
    }
    
    # OpenWakeWord training dependencies
    training_deps = {
        "speechbrain": "speechbrain",
        "torchmetrics": "torchmetrics",
        "mutagen": "mutagen",
        "pronouncing": "pronouncing",
    }
    
    all_deps = {**required, **piper_deps, **training_deps}
    
    missing = []
    for module, package in all_deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("")
        logger.error("Install training dependencies with:")
        logger.error("  pip install -r requirements-training.txt")
        logger.error("")
        logger.error("Or install individually:")
        logger.error(f"  pip install {' '.join(missing)}")
        return False
    
    # Check scipy version (acoustics compatibility)
    try:
        import scipy
        version = tuple(map(int, scipy.__version__.split('.')[:2]))
        if version >= (1, 14):
            logger.warning(f"scipy {scipy.__version__} may have compatibility issues with acoustics")
            logger.warning("If you see errors, run: pip install scipy==1.13.1")
    except Exception:
        pass
    
    return True


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress indicator."""
    logger.info(f"{desc}")
    logger.info(f"  URL: {url}")
    logger.info(f"  Destination: {dest_path}")
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        logger.info(f"  ✓ File already exists, skipping")
        return True
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                downloaded = count * block_size
                percent = min(100, int(downloaded * 100 / total_size))
                mb_downloaded = downloaded // (1024 * 1024)
                mb_total = total_size // (1024 * 1024)
                print(f"\r  Progress: {percent}% ({mb_downloaded}/{mb_total} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, str(dest_path), progress_hook)
        print()  # New line after progress
        logger.info(f"  ✓ Download complete")
        return True
    except Exception as e:
        logger.error(f"  ✗ Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def setup_piper_tts() -> bool:
    """Clone and setup Piper sample generator."""
    logger.info("\n" + "─" * 60)
    logger.info("Setting up Piper TTS Sample Generator")
    logger.info("─" * 60)
    
    piper_models_dir = PIPER_DIR / "models"
    model_file = piper_models_dir / "en-us-libritts-high.pt"
    
    # Check if already fully setup
    if PIPER_DIR.exists() and (PIPER_DIR / "generate_samples.py").exists() and model_file.exists():
        logger.info("  ✓ Piper sample generator already exists")
        return True
    
    # Clone repo if needed
    if not (PIPER_DIR / "generate_samples.py").exists():
        logger.info("  Cloning piper-sample-generator...")
        try:
            if PIPER_DIR.exists():
                shutil.rmtree(PIPER_DIR)
            
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/dscripka/piper-sample-generator.git",
                str(PIPER_DIR)
            ], check=True, capture_output=True)
            
            logger.info("  ✓ Cloned successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Failed to clone Piper: {e}")
            return False
    
    # Download TTS voice model if needed
    if not model_file.exists():
        logger.info("  Downloading Piper TTS voice model (~243 MB)...")
        piper_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Download from rhasspy/piper-sample-generator releases
        model_url = "https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt"
        
        if not download_file(model_url, model_file, "  Piper TTS model (en-us-libritts-high)"):
            logger.error("  ✗ Failed to download Piper TTS model")
            return False
    
    # Install piper-tts package
    logger.info("  Installing piper-tts package...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "piper-tts", "-q"
        ], check=True)
    except subprocess.CalledProcessError:
        logger.warning("  piper-tts may already be installed or failed")
    
    logger.info("  ✓ Piper TTS setup complete")
    return True


def download_mit_rirs() -> Path | None:
    """Download MIT room impulse responses."""
    logger.info("\n" + "─" * 60)
    logger.info("Downloading MIT Room Impulse Responses")
    logger.info("─" * 60)
    
    rir_dir = DATASETS_DIR / "mit_rirs"
    marker_file = rir_dir / ".download_complete"
    
    if marker_file.exists():
        wav_count = len(list(rir_dir.glob("*.wav")))
        logger.info(f"  ✓ MIT RIRs already downloaded ({wav_count} files)")
        return rir_dir
    
    logger.info("  Downloading via HuggingFace datasets...")
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
        
        rir_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the dataset
        ds = load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            trust_remote_code=True
        )
        
        # Save each audio file
        for i, item in enumerate(ds):
            audio = item["audio"]
            path = rir_dir / f"rir_{i:04d}.wav"
            
            # Convert to numpy array and save
            audio_array = np.array(audio["array"], dtype=np.float32)
            sf.write(str(path), audio_array, audio["sampling_rate"])
            
            if (i + 1) % 50 == 0:
                print(f"\r  Saved {i + 1}/{len(ds)} impulse responses", end="", flush=True)
        
        print()
        marker_file.touch()
        logger.info(f"  ✓ Downloaded {len(ds)} impulse responses")
        return rir_dir
        
    except ImportError:
        logger.warning("  datasets or soundfile package not installed")
        logger.warning("  Run: pip install datasets soundfile")
        logger.warning("  Continuing without room impulse responses...")
        return None
    except Exception as e:
        logger.warning(f"  Failed to download MIT RIRs: {e}")
        logger.warning("  Continuing without room impulse responses...")
        return None


def download_validation_features() -> Path | None:
    """Download validation features dataset."""
    logger.info("\n" + "─" * 60)
    logger.info("Downloading Validation Features (~500MB)")
    logger.info("─" * 60)
    
    dest = DATASETS_DIR / DATASETS["validation_features"]["filename"]
    
    if download_file(
        DATASETS["validation_features"]["url"],
        dest,
        "Validation features (required for false positive calibration)"
    ):
        return dest
    return None


def download_negative_features() -> Path | None:
    """Download negative features dataset (large, optional)."""
    logger.info("\n" + "─" * 60)
    logger.info("Negative Features Dataset (~8GB)")
    logger.info("─" * 60)
    
    dest = DATASETS_DIR / DATASETS["negative_features"]["filename"]
    
    if dest.exists():
        logger.info("  ✓ Negative features already downloaded")
        return dest
    
    print()
    print("  The negative features dataset is ~8GB and provides 2000 hours")
    print("  of background audio for training. This significantly improves")
    print("  model accuracy but is optional.")
    print()
    
    response = input("  Download negative features? [y/N]: ").strip().lower()
    
    if response == 'y':
        if download_file(
            DATASETS["negative_features"]["url"],
            dest,
            "Negative features (2000 hours of background audio)"
        ):
            return dest
    else:
        logger.info("  Skipping negative features (model will still work)")
    
    return None


def create_training_config(rir_path: Path | None, val_path: Path | None, neg_path: Path | None) -> Path:
    """Create the YAML configuration file for OpenWakeWord training."""
    import yaml
    
    logger.info("\n" + "─" * 60)
    logger.info("Creating Training Configuration")
    logger.info("─" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "training_config.yaml"
    
    # Build feature data files dict
    feature_data_files = {}
    if neg_path and neg_path.exists():
        feature_data_files["ACAV100M"] = str(neg_path)
    
    # Build batch sizes based on available data
    if neg_path and neg_path.exists():
        batch_n_per_class = {
            "ACAV100M": 512,
            "adversarial_negative": 50,
            "positive": 50,
        }
    else:
        batch_n_per_class = {
            "adversarial_negative": 256,
            "positive": 256,
        }
    
    config = {
        # Basic settings
        "model_name": TRAINING_CONFIG["model_name"],
        "target_phrase": TRAINING_CONFIG["target_phrase"],
        "n_samples": TRAINING_CONFIG["n_samples"],
        "n_samples_val": TRAINING_CONFIG["n_samples_val"],
        "tts_batch_size": TRAINING_CONFIG["tts_batch_size"],
        
        # Paths
        "piper_sample_generator_path": str(PIPER_DIR),
        "output_dir": str(OUTPUT_DIR),
        
        # Room impulse responses for augmentation
        "rir_paths": [str(rir_path)] if rir_path and rir_path.exists() else [],
        
        # Background noise (empty for now)
        "background_paths": [],
        "background_paths_duplication_rate": [],
        
        # Validation data for false positive calibration
        "false_positive_validation_data_path": str(val_path) if val_path and val_path.exists() else "",
        
        # Negative feature data
        "feature_data_files": feature_data_files,
        
        # Augmentation settings
        "augmentation_rounds": TRAINING_CONFIG["augmentation_rounds"],
        "augmentation_batch_size": TRAINING_CONFIG["augmentation_batch_size"],
        
        # Training parameters
        "batch_n_per_class": batch_n_per_class,
        "max_steps": TRAINING_CONFIG["max_steps"],
        "max_negative_weight": TRAINING_CONFIG["max_negative_weight"],
        "target_false_positives_per_hour": TRAINING_CONFIG["target_false_positives_per_hour"],
        "model_type": TRAINING_CONFIG["model_type"],
        
        # Custom negative phrases
        "custom_negative_phrases": TRAINING_CONFIG["custom_negative_phrases"],
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    logger.info(f"  ✓ Config saved to: {config_path}")
    
    # Print summary
    logger.info("\n  Configuration Summary:")
    logger.info(f"    Target phrase: {config['target_phrase']}")
    logger.info(f"    Training samples: {config['n_samples']}")
    logger.info(f"    Validation samples: {config['n_samples_val']}")
    logger.info(f"    RIR augmentation: {'Yes' if config['rir_paths'] else 'No'}")
    logger.info(f"    Negative features: {'ACAV100M (2000hrs)' if feature_data_files else 'Adversarial only'}")
    logger.info(f"    Max training steps: {config['max_steps']}")
    
    return config_path


def run_training_step(step: str) -> bool:
    """Run a training step using OpenWakeWord's train module."""
    
    config_path = OUTPUT_DIR / "training_config.yaml"
    
    if not config_path.exists():
        logger.error("Training config not found. Run --setup first.")
        return False
    
    step_info = {
        "generate": {
            "flag": "--generate_clips",
            "desc": "Generating synthetic audio clips using Piper TTS",
            "time_est": "10-30 minutes",
        },
        "augment": {
            "flag": "--augment_clips",
            "desc": "Augmenting clips with noise and reverb",
            "time_est": "5-15 minutes",
        },
        "train": {
            "flag": "--train_model",
            "desc": "Training the wake word model",
            "time_est": "30-120 minutes",
        },
    }
    
    if step not in step_info:
        logger.error(f"Unknown step: {step}")
        return False
    
    info = step_info[step]
    
    logger.info("\n" + "═" * 60)
    logger.info(f"STEP: {step.upper()}")
    logger.info("═" * 60)
    logger.info(f"  {info['desc']}")
    logger.info(f"  Estimated time: {info['time_est']}")
    logger.info("")
    
    cmd = [
        sys.executable, "-m", "openwakeword.train",
        "--training_config", str(config_path),
        info["flag"]
    ]
    
    # Add overwrite flag for augmentation
    if step == "augment":
        cmd.append("--overwrite")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            print(f"  {line}", end='')
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"  ✗ Step '{step}' failed with code {process.returncode}")
            return False
        
        logger.info(f"\n  ✓ Step '{step}' completed successfully")
        return True
        
    except FileNotFoundError:
        logger.error("  ✗ Could not find openwakeword.train module")
        logger.error("  Make sure openwakeword is installed: pip install openwakeword")
        return False
    except Exception as e:
        logger.error(f"  ✗ Step '{step}' failed: {e}")
        return False


def copy_model_to_client() -> bool:
    """Copy the trained model to the client folder."""
    source = OUTPUT_DIR / "hey_icarus.onnx"
    client_models = BASE_DIR.parent / "Client" / "models"
    dest = client_models / "hey_icarus.onnx"
    
    if not source.exists():
        logger.warning(f"Model not found: {source}")
        return False
    
    client_models.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, dest)
    
    logger.info(f"  ✓ Model copied to: {dest}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline Functions
# ═══════════════════════════════════════════════════════════════════════════════

def run_setup() -> bool:
    """Run the setup phase: download datasets and setup Piper."""
    logger.info("\n" + "═" * 60)
    logger.info("PHASE 1: SETUP")
    logger.info("═" * 60)
    
    if not check_dependencies():
        return False
    
    # Setup Piper TTS
    if not setup_piper_tts():
        return False
    
    # Download MIT RIRs
    rir_path = download_mit_rirs()
    
    # Download validation features
    val_path = download_validation_features()
    
    # Download negative features (optional, user prompted)
    neg_path = download_negative_features()
    
    # Create training config
    config_path = create_training_config(rir_path, val_path, neg_path)
    
    logger.info("\n" + "═" * 60)
    logger.info("SETUP COMPLETE")
    logger.info("═" * 60)
    logger.info(f"\nConfig file: {config_path}")
    logger.info("\nNext steps:")
    logger.info("  python train_wake_word.py --generate")
    logger.info("  python train_wake_word.py --augment")
    logger.info("  python train_wake_word.py --train")
    logger.info("\nOr run all at once:")
    logger.info("  python train_wake_word.py --generate --augment --train")
    
    return True


def run_generate() -> bool:
    """Generate synthetic audio clips."""
    return run_training_step("generate")


def run_augment() -> bool:
    """Augment clips with noise and reverb."""
    return run_training_step("augment")


def run_train() -> bool:
    """Train the wake word model."""
    success = run_training_step("train")
    
    if success:
        # Check if model was created
        model_path = OUTPUT_DIR / "hey_icarus.onnx"
        
        if model_path.exists():
            logger.info("\n" + "═" * 60)
            logger.info("TRAINING COMPLETE!")
            logger.info("═" * 60)
            logger.info(f"\nModel saved to: {model_path}")
            logger.info(f"Model size: {model_path.stat().st_size / 1024:.1f} KB")
            
            # Copy to client
            copy_model_to_client()
            
            logger.info("\n" + "─" * 60)
            logger.info("To use in client.py:")
            logger.info("─" * 60)
            logger.info('\n  # Option 1: Load from bundled models')
            logger.info('  model = openwakeword.Model(wakeword_models=["hey_icarus"])')
            logger.info('\n  # Option 2: Load from file path')
            logger.info('  model = openwakeword.Model(wakeword_models=["models/hey_icarus.onnx"])')
        else:
            logger.warning("Model file not found after training")
    
    return success


def run_all() -> bool:
    """Run the entire pipeline."""
    steps = [
        ("Setup", run_setup),
        ("Generate", run_generate),
        ("Augment", run_augment),
        ("Train", run_train),
    ]
    
    for name, func in steps:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# PIPELINE: {name.upper()}")
        logger.info(f"{'#' * 60}")
        
        if not func():
            logger.error(f"\n✗ Pipeline failed at step: {name}")
            return False
    
    logger.info("\n" + "═" * 60)
    logger.info("FULL PIPELINE COMPLETE!")
    logger.info("═" * 60)
    
    return True


def show_status():
    """Show current training status and file locations."""
    print("\n" + "═" * 60)
    print("TRAINING STATUS")
    print("═" * 60)
    
    checks = [
        ("Piper TTS", PIPER_DIR / "generate_samples.py"),
        ("MIT RIRs", DATASETS_DIR / "mit_rirs" / ".download_complete"),
        ("Validation Features", DATASETS_DIR / "validation_set_features.npy"),
        ("Negative Features", DATASETS_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
        ("Training Config", OUTPUT_DIR / "training_config.yaml"),
        ("Generated Clips", OUTPUT_DIR / "clips"),
        ("Augmented Clips", OUTPUT_DIR / "augmented_clips"),
        ("Trained Model", OUTPUT_DIR / "hey_icarus.onnx"),
    ]
    
    for name, path in checks:
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom 'Hey Icarus' wake word model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_wake_word.py                  # Interactive menu
  python train_wake_word.py --setup          # Download datasets, setup TTS
  python train_wake_word.py --generate       # Generate synthetic clips
  python train_wake_word.py --augment        # Augment with noise/reverb
  python train_wake_word.py --train          # Train the model
  python train_wake_word.py --all            # Run entire pipeline
  python train_wake_word.py --status         # Check training status
        """
    )
    
    parser.add_argument("--setup", action="store_true", 
                        help="Download datasets and setup Piper TTS")
    parser.add_argument("--generate", action="store_true", 
                        help="Generate synthetic audio clips")
    parser.add_argument("--augment", action="store_true", 
                        help="Augment clips with noise/reverb")
    parser.add_argument("--train", action="store_true", 
                        help="Train the wake word model")
    parser.add_argument("--all", action="store_true", 
                        help="Run entire pipeline")
    parser.add_argument("--status", action="store_true", 
                        help="Show current training status")
    
    args = parser.parse_args()
    
    # If no arguments, show interactive menu
    if not any([args.setup, args.generate, args.augment, args.train, args.all, args.status]):
        print_banner()
        
        print("Select an option:")
        print("  1. Run full pipeline (recommended for first time)")
        print("  2. Setup only (download datasets)")
        print("  3. Generate clips only")
        print("  4. Augment clips only")
        print("  5. Train model only")
        print("  6. Show status")
        print("  7. Use pre-built 'hey_jarvis' instead (no training)")
        print("  q. Quit")
        
        choice = input("\nYour choice [1-7/q]: ").strip().lower()
        
        if choice == '1':
            args.all = True
        elif choice == '2':
            args.setup = True
        elif choice == '3':
            args.generate = True
        elif choice == '4':
            args.augment = True
        elif choice == '5':
            args.train = True
        elif choice == '6':
            show_status()
            return
        elif choice == '7':
            print("""
╔══════════════════════════════════════════════════════════════╗
║              Using Pre-built 'hey_jarvis'                    ║
╚══════════════════════════════════════════════════════════════╝

Your client.py is already configured to use 'hey_jarvis'.
Just run: python client.py

Say "Hey Jarvis" to activate. No training required!

To switch to 'Hey Icarus' later, complete the training pipeline
and update client.py to use the 'hey_icarus' model.
            """)
            return
        else:
            print("Cancelled.")
            return
    
    # Run status check
    if args.status:
        show_status()
        return
    
    # Run full pipeline
    if args.all:
        success = run_all()
        sys.exit(0 if success else 1)
    
    # Run individual steps in order
    if args.setup:
        if not run_setup():
            sys.exit(1)
    
    if args.generate:
        if not run_generate():
            sys.exit(1)
    
    if args.augment:
        if not run_augment():
            sys.exit(1)
    
    if args.train:
        if not run_train():
            sys.exit(1)


if __name__ == "__main__":
    main()
