#!/usr/bin/env python3
"""
Script to improve training data for better multimodal LLM performance.
Focuses on creating balanced data for both item recommendation and stop prediction.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

def analyze_current_data():
    """Analyze current training data distribution."""
    print("Analyzing current training data...")
    
    with open('data/processed/train.json', 'r') as f:
        train_data = json.load(f)
    
    # Analyze sequence lengths and EOS patterns
    sequence_lengths = []
    eos_counts = []
    item_counts = []
    
    for entry in train_data[:1000]:  # Sample analysis
        output = entry['output']
        items = output.split(', ')
        eos_count = sum(1 for item in items if item == '<|endoftext|>')
        item_count = len(items) - eos_count
        
        sequence_lengths.append(len(items))
        eos_counts.append(eos_count)
        item_counts.append(item_count)
    
    print(f"Average sequence length: {sum(sequence_lengths)/len(sequence_lengths):.2f}")
    print(f"Average EOS count: {sum(eos_counts)/len(eos_counts):.2f}")
    print(f"Average item count: {sum(item_counts)/len(item_counts):.2f}")
    print(f"Sequences with mostly EOS (≥7): {sum(1 for x in eos_counts if x >= 7)}/{len(eos_counts)}")
    print(f"Sequences with few items (≤2): {sum(1 for x in item_counts if x <= 2)}/{len(item_counts)}")

def create_balanced_dataset():
    """Create a more balanced dataset for training."""
    print("Creating balanced dataset...")
    
    with open('data/processed/train.json', 'r') as f:
        train_data = json.load(f)
    
    # Categorize sequences by their characteristics
    long_sequences = []  # Many items, few EOS
    short_sequences = []  # Few items, many EOS
    balanced_sequences = []  # Mixed
    
    for entry in train_data:
        output = entry['output']
        items = output.split(', ')
        eos_count = sum(1 for item in items if item == '<|endoftext|>')
        item_count = len(items) - eos_count
        
        if item_count >= 5 and eos_count <= 3:
            long_sequences.append(entry)
        elif item_count <= 2 and eos_count >= 7:
            short_sequences.append(entry)
        else:
            balanced_sequences.append(entry)
    
    print(f"Long sequences (many items): {len(long_sequences)}")
    print(f"Short sequences (many EOS): {len(short_sequences)}")
    print(f"Balanced sequences: {len(balanced_sequences)}")
    
    # Create balanced dataset
    # Take equal amounts from each category
    min_size = min(len(long_sequences), len(short_sequences), len(balanced_sequences))
    
    balanced_dataset = []
    balanced_dataset.extend(random.sample(long_sequences, min_size))
    balanced_dataset.extend(random.sample(short_sequences, min_size))
    balanced_dataset.extend(random.sample(balanced_sequences, min_size))
    
    # Shuffle the dataset
    random.shuffle(balanced_dataset)
    
    print(f"Created balanced dataset with {len(balanced_dataset)} sequences")
    
    # Save balanced dataset
    with open('data/processed/train_balanced.json', 'w') as f:
        json.dump(balanced_dataset, f, indent=2)
    
    return balanced_dataset

def create_stop_prediction_examples():
    """Create additional examples to teach stop prediction."""
    print("Creating stop prediction examples...")
    
    with open('data/processed/train.json', 'r') as f:
        train_data = json.load(f)
    
    stop_examples = []
    
    # Find sequences where users stopped early (many EOS tokens)
    for entry in train_data:
        output = entry['output']
        items = output.split(', ')
        eos_count = sum(1 for item in items if item == '<|endoftext|>')
        item_count = len(items) - eos_count
        
        # If user bought few items and then stopped, create multiple examples
        if item_count <= 2 and eos_count >= 6:
            # Create variations with different stopping patterns
            for stop_position in range(item_count + 1, len(items) + 1):
                new_output = items[:stop_position] + ['<|endoftext|>'] * (len(items) - stop_position)
                new_entry = entry.copy()
                new_entry['output'] = ', '.join(new_output)
                stop_examples.append(new_entry)
    
    print(f"Created {len(stop_examples)} stop prediction examples")
    
    # Save stop examples
    with open('data/processed/train_stop_examples.json', 'w') as f:
        json.dump(stop_examples, f, indent=2)
    
    return stop_examples

def main():
    """Main function to improve training data."""
    print("=== Multimodal LLM Training Data Improvement ===")
    
    # Analyze current data
    analyze_current_data()
    
    # Create balanced dataset
    balanced_data = create_balanced_dataset()
    
    # Create stop prediction examples
    stop_examples = create_stop_prediction_examples()
    
    # Combine datasets
    print("Combining datasets...")
    combined_data = balanced_data + stop_examples
    random.shuffle(combined_data)
    
    print(f"Final combined dataset: {len(combined_data)} sequences")
    
    # Save final dataset
    with open('data/processed/train_improved.json', 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print("Improved training data saved to data/processed/train_improved.json")
    print("\nTo use this improved data:")
    print("1. Backup your current train.json")
    print("2. Replace train.json with train_improved.json")
    print("3. Retrain the model")

if __name__ == "__main__":
    main()
