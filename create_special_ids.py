#!/usr/bin/env python3
"""
Smart data preparation script that only creates missing files without re-processing data.
This script checks if data already exists and only creates the special_user_item_ids.json file if missing.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.utils import setup_logging, load_config

def create_special_user_item_ids():
    """Create special_user_item_ids.json from existing data without re-processing."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Checking for existing data files...")
    
    # Check if data files exist
    processed_dir = Path("data/processed")
    train_file = processed_dir / "train.json"
    val_file = processed_dir / "val.json"
    test_file = processed_dir / "test.json"
    special_ids_file = processed_dir / "special_user_item_ids.json"
    
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.info("Please run the full data preparation first.")
        return False
    
    if special_ids_file.exists():
        logger.info(f"special_user_item_ids.json already exists: {special_ids_file}")
        return True
    
    logger.info("Creating special_user_item_ids.json from existing data...")
    
    try:
        # Load training data
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # Extract all unique ASINs
        all_asins = set()
        for entry in train_data:
            if 'asin' in entry:
                all_asins.add(entry['asin'])
            if 'parent_asin' in entry:
                all_asins.add(entry['parent_asin'])
        
        # Create special user item IDs
        special_user_item_ids = ['<|endoftext|>'] + list(all_asins)
        
        # Save to file
        with open(special_ids_file, 'w') as f:
            json.dump(special_user_item_ids, f)
        
        logger.info(f"✅ Created special_user_item_ids.json with {len(special_user_item_ids)} tokens")
        logger.info(f"📁 Saved to: {special_ids_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create special_user_item_ids.json: {e}")
        return False

def main():
    """Main function."""
    print("🔧 Smart Data Preparation - Creating Missing Files Only")
    print("=" * 60)
    
    success = create_special_user_item_ids()
    
    if success:
        print("Data preparation completed successfully!")
        print("special_user_item_ids.json created/verified")
    else:
        print("Data preparation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
