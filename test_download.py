#!/usr/bin/env python3
"""
Test script to verify the multi-category data download functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.multi_category_loader import MultiCategoryDataLoader, download_amazon_categories
from src.utils.utils import load_config, setup_logging
import logging

def test_download():
    """Test downloading data for the specified categories."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Categories to download
    categories = ['Appliances', 'Digital_Music', 'Gift_Cards', 'Health_and_Personal_Care']
    
    logger.info("Testing multi-category data download...")
    logger.info(f"Categories: {categories}")
    
    # Test individual category download
    for category in categories:
        logger.info(f"\nTesting download for {category}...")
        
        try:
            # Create a minimal config
            config = {
                'data_config': {
                    'amazon_categories': [category],
                    'use_multiple_categories': True
                },
                'model_config': {
                    'test_size': 10000
                }
            }
            
            loader = MultiCategoryDataLoader(config)
            review_file, meta_file = loader.download_category_data(category)
            
            logger.info(f"✓ Successfully downloaded {category}")
            logger.info(f"  Review file: {review_file}")
            logger.info(f"  Meta file: {meta_file}")
            
            # Check file sizes
            review_size = Path(review_file).stat().st_size
            meta_size = Path(meta_file).stat().st_size
            
            logger.info(f"  Review file size: {review_size / (1024*1024):.2f} MB")
            logger.info(f"  Meta file size: {meta_size / (1024*1024):.2f} MB")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {category}: {e}")
    
    # Test batch download
    logger.info(f"\nTesting batch download for all categories...")
    try:
        results = download_amazon_categories(categories)
        logger.info(f"Batch download results: {results}")
        
        success_count = sum(results.values())
        logger.info(f"Successfully downloaded {success_count}/{len(categories)} categories")
        
    except Exception as e:
        logger.error(f"Batch download failed: {e}")
    
    logger.info("\nDownload test completed!")

if __name__ == "__main__":
    test_download()


