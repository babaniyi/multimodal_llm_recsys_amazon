"""
Improved data filtering strategies for better multimodal model performance.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def filter_users_by_sequence_length(review_data: List[Dict], min_items_per_sequence: int = 5) -> List[Dict]:
    """
    Filter users to keep only those with sequences that have at least min_items_per_sequence items.
    This ensures the model sees longer sequences during training.
    """
    # Group by user and create sequences
    user_sequences = defaultdict(list)
    for review in review_data:
        user_id = review.get('user_id')
        if user_id:
            user_sequences[user_id].append(review)
    
    # Filter users with long enough sequences
    valid_users = set()
    for user_id, interactions in user_sequences.items():
        if len(interactions) >= min_items_per_sequence:
            valid_users.add(user_id)
    
    # Filter review data
    filtered_data = [review for review in review_data 
                    if review.get('user_id') in valid_users]
    
    logger.info(f"Filtered to {len(filtered_data)} reviews from {len(valid_users)} users with {min_items_per_sequence}+ items per sequence")
    return filtered_data

def create_longer_sequences(df: pd.DataFrame, min_sequence_length: int = 8) -> pd.DataFrame:
    """
    Create longer sequences by combining multiple user interactions.
    This helps the model learn to predict more items before stopping.
    """
    df_sorted = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    enhanced_data = []
    
    for user_id, user_data in df_sorted.groupby('user_id'):
        user_data = user_data.reset_index(drop=True)
        
        if len(user_data) < min_sequence_length:
            continue
            
        # Create sliding window sequences
        for i in range(len(user_data) - min_sequence_length + 1):
            sequence = user_data.iloc[i:i + min_sequence_length]
            
            # Create input sequence (first 70% of items)
            input_length = int(len(sequence) * 0.7)
            input_sequence = sequence.iloc[:input_length]
            
            # Create target sequence (remaining 30% of items)
            target_sequence = sequence.iloc[input_length:]
            
            # Add to enhanced data
            enhanced_data.append({
                'user_id': user_id,
                'input_sequence': input_sequence['parent_asin'].tolist(),
                'target_sequence': target_sequence['parent_asin'].tolist(),
                'input_titles': input_sequence['title'].tolist(),
                'target_titles': target_sequence['title'].tolist(),
                'timestamp': sequence.iloc[0]['timestamp']
            })
    
    return pd.DataFrame(enhanced_data)

def balance_sequence_lengths(df: pd.DataFrame, target_length: int = 10) -> pd.DataFrame:
    """
    Balance the dataset to have more sequences of target length.
    """
    # Analyze current sequence length distribution
    sequence_lengths = df.groupby('user_id').size()
    
    # Create balanced sampling
    balanced_data = []
    
    for length in range(5, target_length + 1):
        users_with_length = sequence_lengths[sequence_lengths >= length].index
        sample_size = min(len(users_with_length), 1000)  # Limit per length
        
        if sample_size > 0:
            sampled_users = np.random.choice(users_with_length, sample_size, replace=False)
            for user_id in sampled_users:
                user_data = df[df['user_id'] == user_id].sort_values('timestamp')
                # Take first 'length' items
                sequence = user_data.head(length)
                balanced_data.append({
                    'user_id': user_id,
                    'sequence': sequence['parent_asin'].tolist(),
                    'titles': sequence['title'].tolist(),
                    'length': length
                })
    
    return pd.DataFrame(balanced_data)

def create_contrastive_samples(df: pd.DataFrame, num_negative_samples: int = 3) -> pd.DataFrame:
    """
    Create contrastive learning samples with positive and negative examples.
    """
    all_items = set(df['parent_asin'].unique())
    contrastive_data = []
    
    for user_id, user_data in df.groupby('user_id'):
        user_items = set(user_data['parent_asin'].unique())
        negative_items = list(all_items - user_items)
        
        for _, interaction in user_data.iterrows():
            # Positive sample
            contrastive_data.append({
                'user_id': user_id,
                'item_id': interaction['parent_asin'],
                'label': 1,
                'title': interaction['title'],
                'review_text': interaction.get('review_text', '')
            })
            
            # Negative samples
            if negative_items:
                negative_sample = np.random.choice(negative_items, 
                                                  min(num_negative_samples, len(negative_items)), 
                                                  replace=False)
                for neg_item in negative_sample:
                    contrastive_data.append({
                        'user_id': user_id,
                        'item_id': neg_item,
                        'label': 0,
                        'title': '',  # Will be filled from metadata
                        'review_text': ''
                    })
    
    return pd.DataFrame(contrastive_data)

def main():
    """
    Example usage of improved data filtering strategies.
    """
    logger.info("Applying improved data filtering strategies...")
    
    # Load your existing data
    # df = pd.read_json('data/processed/processed_data.jsonl', lines=True)
    
    # Apply strategies
    # 1. Filter for longer sequences
    # filtered_df = filter_users_by_sequence_length(df, min_items_per_sequence=8)
    
    # 2. Create longer sequences
    # enhanced_df = create_longer_sequences(filtered_df, min_sequence_length=10)
    
    # 3. Balance sequence lengths
    # balanced_df = balance_sequence_lengths(enhanced_df, target_length=10)
    
    # 4. Create contrastive samples
    # contrastive_df = create_contrastive_samples(balanced_df, num_negative_samples=3)
    
    logger.info("Data filtering strategies applied successfully!")

if __name__ == "__main__":
    main()
