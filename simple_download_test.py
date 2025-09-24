#!/usr/bin/env python3
"""
Simple test script to verify the download URLs work.
"""

import requests
import os
from pathlib import Path

def test_download_urls():
    """Test if the download URLs are accessible."""
    
    # Categories to test
    categories = ['Appliances', 'Digital_Music', 'Gift_Cards', 'Health_and_Personal_Care']
    
    base_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    
    print("Testing Amazon Reviews 2023 download URLs...")
    print("=" * 60)
    
    for category in categories:
        print(f"\nTesting {category}...")
        
        # Test review URL
        review_url = f"{base_url}review_categories/{category}.jsonl.gz"
        print(f"Review URL: {review_url}")
        
        try:
            response = requests.head(review_url, timeout=10)
            if response.status_code == 200:
                print(f"✓ Review URL accessible (Status: {response.status_code})")
                print(f"  Content-Length: {response.headers.get('content-length', 'Unknown')} bytes")
            else:
                print(f"✗ Review URL failed (Status: {response.status_code})")
        except Exception as e:
            print(f"✗ Review URL error: {e}")
        
        # Test meta URL
        meta_url = f"{base_url}meta_categories/meta_{category}.jsonl.gz"
        print(f"Meta URL: {meta_url}")
        
        try:
            response = requests.head(meta_url, timeout=10)
            if response.status_code == 200:
                print(f"✓ Meta URL accessible (Status: {response.status_code})")
                print(f"  Content-Length: {response.headers.get('content-length', 'Unknown')} bytes")
            else:
                print(f"✗ Meta URL failed (Status: {response.status_code})")
        except Exception as e:
            print(f"✗ Meta URL error: {e}")
    
    print("\n" + "=" * 60)
    print("URL test completed!")

if __name__ == "__main__":
    test_download_urls()


