import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.metrics import precision_score, recall_score, ndcg_score
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

def calculate_metrics(predictions: List[List[int]], targets: List[List[int]], k: int = 10) -> Dict[str, float]:
    """
    Calculate various recommendation metrics.
    
    Args:
        predictions: List of predicted item lists
        targets: List of target item lists
        k: Number of items to consider for top-k metrics
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Calculate Hit Rate@k (treating <|endoftext|> as valid predictions)
    hits = 0
    total = len(predictions)
    for pred, target in zip(predictions, targets):
        # Check if any predicted item (including <|endoftext|>) matches target
        if any(item in target for item in pred[:k]):
            hits += 1
    metrics[f'hr@{k}'] = hits / total
    
    # Calculate Precision@k (treating <|endoftext|> as valid predictions)
    total_precision = 0
    for pred, target in zip(predictions, targets):
        hits = len(set(pred[:k]) & set(target))
        total_precision += hits / k
    metrics[f'precision@{k}'] = total_precision / total
    
    # Calculate Recall@k (treating <|endoftext|> as valid predictions)
    total_recall = 0
    for pred, target in zip(predictions, targets):
        hits = len(set(pred[:k]) & set(target))
        total_recall += hits / len(target)
    metrics[f'recall@{k}'] = total_recall / total
    
    # Calculate NDCG@k (treating <|endoftext|> as valid predictions)
    total_ndcg = 0
    for pred, target in zip(predictions, targets):
        dcg = 0
        idcg = 1  # Since we're dealing with binary relevance
        for i, item in enumerate(pred[:k]):
            if item in target:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
        total_ndcg += dcg / idcg
    metrics[f'ndcg@{k}'] = total_ndcg / total
    
    # Calculate MRR (Mean Reciprocal Rank) - treating <|endoftext|> as valid predictions
    total_mrr = 0
    for pred, target in zip(predictions, targets):
        for i, item in enumerate(pred):
            if item in target:
                total_mrr += 1 / (i + 1)
                break
    metrics['mrr'] = total_mrr / total
    
    # Calculate MAP (Mean Average Precision) - treating <|endoftext|> as valid predictions
    total_map = 0
    for pred, target in zip(predictions, targets):
        ap = 0
        hits = 0
        for i, item in enumerate(pred):
            if item in target:
                hits += 1
                ap += hits / (i + 1)
        if hits > 0:
            ap /= hits
        total_map += ap
    metrics['map'] = total_map / total
    
    # Calculate Coverage
    all_items = set()
    recommended_items = set()
    for pred in predictions:
        recommended_items.update(pred[:k])
    for target in targets:
        all_items.update(target)
    metrics['coverage'] = len(recommended_items) / len(all_items)
    
    # Calculate Diversity (using Jaccard similarity)
    total_diversity = 0
    count = 0
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            pred1 = set(predictions[i][:k])
            pred2 = set(predictions[j][:k])
            intersection = len(pred1 & pred2)
            union = len(pred1 | pred2)
            if union > 0:
                total_diversity += 1 - (intersection / union)
                count += 1
    metrics['diversity'] = total_diversity / count if count > 0 else 0
    
    # Calculate Novelty (using popularity-based approach)
    item_counts = {}
    for pred in predictions:
        for item in pred[:k]:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    total_novelty = 0
    for pred in predictions:
        pred_novelty = 0
        for item in pred[:k]:
            pred_novelty += 1 / (item_counts[item] + 1)  # +1 to avoid division by zero
        total_novelty += pred_novelty / k
    metrics['novelty'] = total_novelty / len(predictions)
    
    return metrics

# Load the JSON data
with open('data/processed/test_with_responses_processed.json', 'r') as file:
    data = json.load(file)

# Extract predictions and targets
predictions = [entry['model_response'].split(', ') for entry in data]
targets = [entry['output'].split(', ') for entry in data]

# Calculate metrics
metrics = calculate_metrics(predictions, targets, k=7)

# Log and print metrics
for metric_name, value in metrics.items():
    logger.info(f"{metric_name}: {value:.4f}")
    print(f"{metric_name}: {value:.4f}") 