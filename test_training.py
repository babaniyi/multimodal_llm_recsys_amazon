#!/usr/bin/env python3
"""
Simple test script to verify training works without CUDA issues.
"""

import os
import sys
import torch

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("Testing training configuration...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Test basic tensor operations
try:
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    z = torch.mm(x, y)
    print("✅ Basic tensor operations work")
except Exception as e:
    print(f"❌ Tensor operations failed: {e}")
    sys.exit(1)

# Test DataLoader
try:
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.randn(100, 10))
    loader = DataLoader(dataset, batch_size=16, num_workers=0)
    batch = next(iter(loader))
    print("✅ DataLoader works")
except Exception as e:
    print(f"❌ DataLoader failed: {e}")
    sys.exit(1)

print("✅ All tests passed! Training should work with CPU-only mode.")

