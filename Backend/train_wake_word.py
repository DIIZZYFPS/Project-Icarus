#!/usr/bin/env python3
"""
Train a custom wake word model for "Hey Icarus" using OpenWakeWord.

This script generates synthetic audio samples using TTS and trains
a wake word detection model. Run on the GPU server for faster training.

Usage:
    python train_wake_word.py

Output:
    models/hey_icarus.onnx - Copy this to Client folder for detection
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WakeWordTrainer")

def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import openwakeword
    except ImportError:
        missing.append("openwakeword")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)
    
    return True


def train_hey_icarus():
    """Train the 'Hey Icarus' wake word model."""
    
    check_dependencies()
    
    from openwakeword.train import train_custom_model
    
    # Create output directory
    os.makedirs("models", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Training 'Hey Icarus' Wake Word Model")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This will:")
    logger.info("  1. Generate synthetic 'hey icarus' audio using TTS")
    logger.info("  2. Generate negative samples (random speech)")
    logger.info("  3. Train a small neural network for detection")
    logger.info("")
    logger.info("Estimated time: 10-30 minutes on GPU")
    logger.info("")
    
    try:
        # Train the custom wake word model
        # This uses OpenWakeWord's built-in synthetic data generation
        train_custom_model(
            # The wake word phrase
            target_phrase="hey icarus",
            
            # Output path for the trained model
            output_path="models/hey_icarus.onnx",
            
            # Number of synthetic samples to generate
            n_samples=5000,
            
            # Training epochs
            epochs=100,
            
            # Batch size (adjust based on GPU memory)
            batch_size=64,
            
            # Use GPU if available
            device="cuda",
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Model saved to: models/hey_icarus.onnx")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Copy models/hey_icarus.onnx to Client folder")
        logger.info("  2. Update client.py to use 'hey_icarus' model")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("")
        logger.info("Alternative: Use the manual training approach")
        logger.info("See: https://github.com/dscripka/openWakeWord/blob/main/docs/training_models.md")
        raise


def main():
    """Main entry point."""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Hey Icarus - Wake Word Training Script              ║
╠══════════════════════════════════════════════════════════════╣
║  This script trains a custom wake word model for Icarus.     ║
║                                                              ║
║  Requirements:                                               ║
║    - NVIDIA GPU (recommended)                                ║
║    - openwakeword package                                    ║
║    - ~30 minutes training time                               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    response = input("Start training? [y/N]: ").strip().lower()
    
    if response == 'y':
        train_hey_icarus()
    else:
        print("Training cancelled.")


if __name__ == "__main__":
    main()
