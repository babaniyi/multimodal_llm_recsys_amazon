#!/usr/bin/env python3
"""
Script to compare model performance before and after improvements.
"""

import json
import os
from pathlib import Path

def load_results(file_path):
    """Load results from JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def compare_metrics(old_results, new_results, metric_name):
    """Compare a specific metric between old and new results."""
    if not old_results or not new_results:
        return "N/A"
    
    old_val = old_results.get(metric_name, {}).get('mean', 0)
    new_val = new_results.get(metric_name, {}).get('mean', 0)
    
    if old_val == 0:
        improvement = "∞" if new_val > 0 else "0%"
    else:
        improvement = f"{((new_val - old_val) / old_val) * 100:.1f}%"
    
    return f"{old_val:.4f} → {new_val:.4f} ({improvement})"

def main():
    """Compare before and after results."""
    print("=== MODEL PERFORMANCE COMPARISON ===\n")
    
    # Load old results (if available)
    old_results = load_results('data/processed/result_metrics_backup.json')
    
    # Load new results
    new_results = load_results('data/processed/result_metrics.json')
    
    if not new_results:
        print("No new results found. Run evaluation first!")
        return
    
    print("KEY METRICS COMPARISON:")
    print("=" * 60)
    
    # Track A: Item Recommendation Quality
    print("\nTRACK A: ITEM RECOMMENDATION QUALITY")
    print("-" * 40)
    
    metrics_track_a = [
        'item_precision@10',
        'item_recall@10', 
        'item_hitrate@10',
        'item_ndcg@10',
        'item_mrr',
        'item_map'
    ]
    
    for metric in metrics_track_a:
        comparison = compare_metrics(old_results, new_results, metric)
        print(f"{metric:20}: {comparison}")
    
    # Track B: Stop Prediction Quality
    print("\nTRACK B: STOP PREDICTION QUALITY")
    print("-" * 40)
    
    metrics_track_b = [
        'stop_hit@10',
        'stop_ndcg@10',
        'stop_prediction_rate'
    ]
    
    for metric in metrics_track_b:
        comparison = compare_metrics(old_results, new_results, metric)
        print(f"{metric:20}: {comparison}")
    
    # Overall Assessment
    print("\nOVERALL ASSESSMENT")
    print("-" * 40)
    
    if new_results.get('item_hitrate@10', {}).get('mean', 0) > 0.01:
        print("Item recommendation: IMPROVED")
    else:
        print("Item recommendation: Still needs work")
    
    if new_results.get('stop_hit@10', {}).get('mean', 0) > 0.1:
        print("Stop prediction: IMPROVED")
    else:
        print("Stop prediction: Still needs work")
    
    # Recommendations
    print("\nNEXT STEPS:")
    print("-" * 40)
    
    item_hit = new_results.get('item_hitrate@10', {}).get('mean', 0)
    stop_hit = new_results.get('stop_hit@10', {}).get('mean', 0)
    
    if item_hit < 0.01 and stop_hit < 0.1:
        print("1. Model Architecture: Consider adding separate heads for items vs stops")
        print("2. Data Quality: Need more diverse training examples")
        print("3. Hyperparameters: Try different learning rates and batch sizes")
    elif item_hit < 0.01:
        print("1. Focus on Item Recommendation: Model needs better item understanding")
        print("2. Analyze what items the model is predicting vs ground truth")
    elif stop_hit < 0.1:
        print("1. Focus on Stop Prediction: Increase stop prediction weight")
        print("2. Add more stop examples to training data")
    else:
        print("1. Great progress! Consider Phase 2 improvements")
        print("2. Fine-tune hyperparameters for further gains")
    
    # Save comparison for future reference
    comparison_data = {
        'timestamp': str(Path().cwd()),
        'old_results': old_results,
        'new_results': new_results,
        'improvements': {
            'item_hitrate': compare_metrics(old_results, new_results, 'item_hitrate@10'),
            'stop_hit': compare_metrics(old_results, new_results, 'stop_hit@10'),
            'item_precision': compare_metrics(old_results, new_results, 'item_precision@10'),
            'stop_ndcg': compare_metrics(old_results, new_results, 'stop_ndcg@10')
        }
    }
    
    with open('data/processed/performance_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison saved to: data/processed/performance_comparison.json")

if __name__ == "__main__":
    main()
