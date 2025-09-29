#!/usr/bin/env python3
"""
Detailed analysis of model improvements focusing on stop prediction vs item recommendation.
"""

import json
import numpy as np
from collections import Counter

def analyze_model_outputs():
    """Analyze the actual model outputs to understand improvements."""
    print("=== DETAILED MODEL OUTPUT ANALYSIS ===\n")
    
    # Load processed test data
    with open('data/processed/test_with_responses_processed.json', 'r') as f:
        test_data = json.load(f)
    
    print(f"Analyzing {len(test_data)} test examples...\n")
    
    # Analyze prediction patterns
    prediction_lengths = []
    eos_positions = []
    item_counts = []
    
    for entry in test_data[:1000]:  # Sample analysis
        model_response = entry['model_response']
        ground_truth = entry['output']
        
        # Parse predictions
        pred_items = model_response.split(', ')
        gt_items = ground_truth.split(', ')
        
        # Count items vs EOS
        pred_item_count = sum(1 for item in pred_items if item != '<|endoftext|>')
        gt_item_count = sum(1 for item in gt_items if item != '<|endoftext|>')
        
        # Find first EOS position
        pred_eos_pos = next((i for i, item in enumerate(pred_items) if item == '<|endoftext|>'), None)
        gt_eos_pos = next((i for i, item in enumerate(gt_items) if item == '<|endoftext|>'), None)
        
        prediction_lengths.append(len(pred_items))
        eos_positions.append(pred_eos_pos)
        item_counts.append(pred_item_count)
    
    # Statistics
    print("📊 PREDICTION STATISTICS:")
    print("-" * 30)
    print(f"Average prediction length: {np.mean(prediction_lengths):.2f}")
    print(f"Average items predicted: {np.mean(item_counts):.2f}")
    print(f"Predictions with EOS: {sum(1 for x in eos_positions if x is not None)}/{len(eos_positions)}")
    
    # EOS position distribution
    eos_positions_valid = [x for x in eos_positions if x is not None]
    if eos_positions_valid:
        print(f"Average EOS position: {np.mean(eos_positions_valid):.2f}")
        print(f"EOS position std: {np.std(eos_positions_valid):.2f}")
    
    # Analyze improvement patterns
    print("\n🔍 IMPROVEMENT PATTERNS:")
    print("-" * 30)
    
    # Check if model is learning to stop earlier
    early_stops = sum(1 for x in eos_positions_valid if x is not None and x <= 5)
    print(f"Early stops (≤5 positions): {early_stops}/{len(eos_positions_valid)}")
    
    # Check item prediction diversity
    all_predicted_items = []
    for entry in test_data[:1000]:
        model_response = entry['model_response']
        pred_items = [item for item in model_response.split(', ') if item != '<|endoftext|>']
        all_predicted_items.extend(pred_items)
    
    unique_items = len(set(all_predicted_items))
    total_items = len(all_predicted_items)
    print(f"Unique items predicted: {unique_items}")
    print(f"Total items predicted: {total_items}")
    print(f"Diversity ratio: {unique_items/total_items:.3f}")

def analyze_ground_truth_patterns():
    """Analyze ground truth patterns to understand the task better."""
    print("\n=== GROUND TRUTH PATTERN ANALYSIS ===\n")
    
    with open('data/processed/test_with_responses_processed.json', 'r') as f:
        test_data = json.load(f)
    
    # Analyze ground truth patterns
    gt_lengths = []
    gt_item_counts = []
    gt_eos_positions = []
    
    for entry in test_data[:1000]:
        ground_truth = entry['output']
        gt_items = ground_truth.split(', ')
        
        gt_item_count = sum(1 for item in gt_items if item != '<|endoftext|>')
        gt_eos_pos = next((i for i, item in enumerate(gt_items) if item == '<|endoftext|>'), None)
        
        gt_lengths.append(len(gt_items))
        gt_item_counts.append(gt_item_count)
        gt_eos_positions.append(gt_eos_pos)
    
    print("📊 GROUND TRUTH STATISTICS:")
    print("-" * 30)
    print(f"Average GT length: {np.mean(gt_lengths):.2f}")
    print(f"Average GT items: {np.mean(gt_item_counts):.2f}")
    print(f"GT with EOS: {sum(1 for x in gt_eos_positions if x is not None)}/{len(gt_eos_positions)}")
    
    # EOS position distribution in ground truth
    gt_eos_positions_valid = [x for x in gt_eos_positions if x is not None]
    if gt_eos_positions_valid:
        print(f"Average GT EOS position: {np.mean(gt_eos_positions_valid):.2f}")
    
    # Task difficulty analysis
    print("\n🎯 TASK DIFFICULTY ANALYSIS:")
    print("-" * 30)
    
    # How many users actually buy multiple items?
    multi_item_users = sum(1 for x in gt_item_counts if x > 1)
    print(f"Users buying multiple items: {multi_item_users}/{len(gt_item_counts)} ({multi_item_users/len(gt_item_counts)*100:.1f}%)")
    
    # How many users stop early?
    early_stop_users = sum(1 for x in gt_eos_positions_valid if x <= 3)
    print(f"Users stopping early (≤3 items): {early_stop_users}/{len(gt_eos_positions_valid)} ({early_stop_users/len(gt_eos_positions_valid)*100:.1f}%)")

def main():
    """Run detailed analysis."""
    analyze_model_outputs()
    analyze_ground_truth_patterns()
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 30)
    print("1. Check if model is predicting fewer items (learning to stop)")
    print("2. Check if EOS positions are getting closer to ground truth")
    print("3. Check if item diversity is improving")
    print("4. Compare prediction patterns with ground truth patterns")

if __name__ == "__main__":
    main()
