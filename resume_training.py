#!/usr/bin/env python3
"""
Script to resume training from a checkpoint.
Useful when training is interrupted due to GPU expiration or other issues.
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.train import main as train_main
from src.utils.utils import setup_logging

def resume_training(checkpoint_path, config_path="configs/train_config.json"):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        config_path (str): Path to the configuration file
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return False
    
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    print("This will continue training from where it left off.")
    print("All training state (model weights, optimizer state, losses) will be restored.")
    
    # Set environment variable to indicate we're resuming
    os.environ['RESUME_TRAINING'] = '1'
    os.environ['CHECKPOINT_PATH'] = str(checkpoint_path)
    
    # Run the training script
    try:
        train_main()
        return True
    except Exception as e:
        print(f"Error resuming training: {e}")
        return False

def list_checkpoints(checkpoint_dir="output/model"):
    """
    List available checkpoints.
    
    Args:
        checkpoint_dir (str): Directory to search for checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print("No checkpoint files found.")
        return
    
    print("Available checkpoints:")
    for i, checkpoint in enumerate(checkpoints, 1):
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"{i}. {checkpoint.name} ({size_mb:.1f} MB)")
    
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument("checkpoint", nargs="?", 
                       help="Path to checkpoint file")
    parser.add_argument("--list", action="store_true",
                       help="List available checkpoints")
    parser.add_argument("--checkpoint-dir", default="output/model",
                       help="Directory to search for checkpoints")
    parser.add_argument("--config", default="configs/train_config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.checkpoint_dir)
        return
    
    if not args.checkpoint:
        # Interactive mode
        print("Training Resume Tool")
        print("===================")
        
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            return
        
        try:
            choice = int(input("\nEnter checkpoint number to resume from: ")) - 1
            if 0 <= choice < len(checkpoints):
                checkpoint_path = checkpoints[choice]
            else:
                print("Invalid choice.")
                return
        except (ValueError, IndexError):
            print("Invalid input.")
            return
    else:
        checkpoint_path = args.checkpoint
    
    # Resume training
    success = resume_training(checkpoint_path, args.config)
    
    if success:
        print("\nTraining resumed successfully!")
    else:
        print("\nFailed to resume training.")

if __name__ == "__main__":
    setup_logging()
    main() 