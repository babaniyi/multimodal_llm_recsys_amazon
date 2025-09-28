"""
Two-track evaluation system for multimodal LLM recommendations.
Separates item recommendation quality from stopping behavior (EOS prediction).

Based on the recommendation to evaluate:
1) Item-only metrics (exclude EOS) - standard recommender metrics
2) Stop metrics (for EOS correctness) - when to stop recommending
"""

import torch
import numpy as np
import math
from typing import List, Dict, Tuple, Union
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# Define EOS token
EOS = "<|endoftext|>"

def split_items(seq: List[str]) -> Tuple[List[str], int]:
    """
    Split sequence into items and find first EOS rank.
    
    Args:
        seq: List of items (can include EOS tokens)
        
    Returns:
        Tuple of (items_only, first_eos_rank)
        - items_only: List without EOS tokens
        - first_eos_rank: Rank (1-indexed) of first EOS, None if no EOS
    """
    items = [x for x in seq if x != EOS]
    eos_ranks = [i+1 for i, x in enumerate(seq) if x == EOS]
    first_eos_rank = eos_ranks[0] if eos_ranks else None
    return items, first_eos_rank

def precision_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """Calculate precision@k for items only (excluding EOS)."""
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    hits = len(set(p_items) & set(t_items))
    return hits / k

def recall_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """Calculate recall@k for items only (excluding EOS)."""
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    if len(t_items) == 0:
        return 0.0
    hits = len(set(p_items) & set(t_items))
    return hits / len(t_items)

def hitrate_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """Calculate hit rate@k for items only (excluding EOS)."""
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    return 1.0 if len(set(p_items) & set(t_items)) > 0 else 0.0

def ndcg_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """Calculate NDCG@k for items only (excluding EOS)."""
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    
<<<<<<< HEAD
    # Calculate Hit Rate@k (treating <|endoftext|> as valid predictions)
    hits = 0
    total = len(predictions)
    for pred, target in zip(predictions, targets):
        # Check if any predicted item (including <|endoftext|>) matches target
        if any(item in target for item in pred[:k]):
=======
    if len(t_items) == 0:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(p_items):
        if item in t_items:
            dcg += 1.0 / math.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(len(t_items), k)):
        idcg += 1.0 / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def mrr_items(pred: List[str], truth: List[str]) -> float:
    """Calculate MRR for items only (excluding EOS)."""
    p_items, _ = split_items(pred)
    t_items, _ = split_items(truth)
    
    if len(t_items) == 0:
        return 0.0
    
    for i, item in enumerate(p_items):
        if item in t_items:
            return 1.0 / (i + 1)
    return 0.0

def map_items(pred: List[str], truth: List[str]) -> float:
    """Calculate MAP for items only (excluding EOS)."""
    p_items, _ = split_items(pred)
    t_items, _ = split_items(truth)
    
    if len(t_items) == 0:
        return 0.0
    
    ap = 0.0
    hits = 0
    for i, item in enumerate(p_items):
        if item in t_items:
>>>>>>> 404062a (improving evaluation metrics with two tracks)
            hits += 1
            ap += hits / (i + 1)
    
<<<<<<< HEAD
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
=======
    return ap / len(t_items) if len(t_items) > 0 else 0.0

def stop_hit_at_k(pred: List[str], truth: List[str], k: int = 10) -> float:
    """Calculate StopHit@k - whether model correctly predicts EOS when ground truth has EOS."""
    _, t_first_eos = split_items(truth[:k])
    _, p_first_eos = split_items(pred[:k])
    return 1.0 if (t_first_eos is not None and p_first_eos is not None) else 0.0

def stop_rank(pred: List[str], k: int = 10) -> Union[int, None]:
    """Calculate StopRank - rank of first EOS prediction."""
    _, p_first_eos = split_items(pred[:k])
    return p_first_eos

def stop_ndcg_at_k(pred: List[str], truth: List[str], k: int = 10) -> float:
    """Calculate Stop-nDCG@k - position-sensitive metric for EOS prediction."""
    _, t_first_eos = split_items(truth[:k])
    _, p_first_eos = split_items(pred[:k])
    
    if t_first_eos is None:
        return 0.0
    
    ideal = 1.0 / math.log2(1 + t_first_eos)
    dcg = 0.0 if p_first_eos is None else 1.0 / math.log2(1 + p_first_eos)
    
    return dcg / ideal if ideal > 0 else 0.0

def coverage_items(predictions: List[List[str]], targets: List[List[str]], k: int = 10) -> float:
    """Calculate coverage for items only (excluding EOS)."""
>>>>>>> 404062a (improving evaluation metrics with two tracks)
    all_items = set()
    recommended_items = set()
    
    for pred in predictions:
        p_items, _ = split_items(pred[:k])
        recommended_items.update(p_items)
    
    for target in targets:
        t_items, _ = split_items(target[:k])
        all_items.update(t_items)
    
    return len(recommended_items) / len(all_items) if len(all_items) > 0 else 0.0

def diversity_items(predictions: List[List[str]], k: int = 10) -> float:
    """Calculate diversity for items only (excluding EOS)."""
    total_diversity = 0.0
    count = 0
    
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            p1_items, _ = split_items(predictions[i][:k])
            p2_items, _ = split_items(predictions[j][:k])
            
            pred1 = set(p1_items)
            pred2 = set(p2_items)
            
            intersection = len(pred1 & pred2)
            union = len(pred1 | pred2)
            
            if union > 0:
                total_diversity += 1 - (intersection / union)
                count += 1
    
    return total_diversity / count if count > 0 else 0.0

def novelty_items(predictions: List[List[str]], k: int = 10) -> float:
    """Calculate novelty for items only (excluding EOS)."""
    item_counts = {}
    
    for pred in predictions:
        p_items, _ = split_items(pred[:k])
        for item in p_items:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    total_novelty = 0.0
    for pred in predictions:
        p_items, _ = split_items(pred[:k])
        pred_novelty = 0.0
        for item in p_items:
            pred_novelty += 1 / (item_counts[item] + 1)
        total_novelty += pred_novelty / max(1, len(p_items))
    
    return total_novelty / len(predictions)

def calculate_two_track_metrics(predictions: List[List[str]], targets: List[List[str]], k: int = 10) -> Dict[str, float]:
    """Calculate comprehensive two-track metrics."""
    metrics = {}
    
    # Track A: Item-only metrics (exclude EOS)
    metrics[f'item_precision@{k}'] = np.mean([precision_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'item_recall@{k}'] = np.mean([recall_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'item_hitrate@{k}'] = np.mean([hitrate_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'item_ndcg@{k}'] = np.mean([ndcg_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics['item_mrr'] = np.mean([mrr_items(pred, truth) for pred, truth in zip(predictions, targets)])
    metrics['item_map'] = np.mean([map_items(pred, truth) for pred, truth in zip(predictions, targets)])
    metrics[f'item_coverage@{k}'] = coverage_items(predictions, targets, k)
    metrics[f'item_diversity@{k}'] = diversity_items(predictions, k)
    metrics[f'item_novelty@{k}'] = novelty_items(predictions, k)
    
    # Track B: Stop metrics (for EOS correctness)
    metrics[f'stop_hit@{k}'] = np.mean([stop_hit_at_k(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'stop_ndcg@{k}'] = np.mean([stop_ndcg_at_k(pred, truth, k) for pred, truth in zip(predictions, targets)])
    
    # Calculate stop rank statistics
    stop_ranks = [stop_rank(pred, k) for pred in predictions]
    stop_ranks = [r for r in stop_ranks if r is not None]
    
    if stop_ranks:
        metrics['stop_rank_median'] = np.median(stop_ranks)
        metrics['stop_rank_mean'] = np.mean(stop_ranks)
        metrics['stop_rank_std'] = np.std(stop_ranks)
        metrics['stop_prediction_rate'] = len(stop_ranks) / len(predictions)
    else:
        metrics['stop_rank_median'] = None
        metrics['stop_rank_mean'] = None
        metrics['stop_rank_std'] = None
        metrics['stop_prediction_rate'] = 0.0
    
    return metrics

def calculate_metrics(predictions: List[List[Union[str, int]]], targets: List[List[Union[str, int]]], k: int = 10) -> Dict[str, float]:
    """
    Main function to calculate two-track metrics.
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        k: Number of items to consider
        
    Returns:
        Dictionary with comprehensive two-track metrics
    """
    logger.info(f"Calculating two-track metrics for {len(predictions)} predictions with k={k}")
    
    # Convert to string format
    str_predictions = [[str(item) for item in pred] for pred in predictions]
    str_targets = [[str(item) for item in target] for target in targets]
    
    return calculate_two_track_metrics(str_predictions, str_targets, k)

if __name__ == "__main__":
    from src.utils.utils import setup_logging
    
    # Setup logging
    setup_logging()
    
    # Test the new metrics with example data
    test_predictions = [
        ["<|ASIN_B0081E9HRY|>", "<|ASIN_B07CP1KY9M|>", "<|ASIN_B07MWVCVR4|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS],
        ["<|ASIN_B07FTFD1XB|>", "<|ASIN_B07CP1KY9M|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS]
    ]
    
    test_targets = [
        ["<|ASIN_B07FTFD1XB|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS],
        ["<|ASIN_B07FTFD1XB|>", "<|ASIN_B07CP1KY9M|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS]
    ]
    
    metrics = calculate_two_track_metrics(test_predictions, test_targets, k=10)
    
    logger.info("Two-Track Evaluation Results:")
    logger.info("=" * 50)
    logger.info("Track A: Item Recommendation Quality (exclude EOS)")
    logger.info(f"  Item Precision@10: {metrics['item_precision@10']:.4f}")
    logger.info(f"  Item Recall@10: {metrics['item_recall@10']:.4f}")
    logger.info(f"  Item Hit Rate@10: {metrics['item_hitrate@10']:.4f}")
    logger.info(f"  Item NDCG@10: {metrics['item_ndcg@10']:.4f}")
    logger.info(f"  Item MRR: {metrics['item_mrr']:.4f}")
    logger.info(f"  Item MAP: {metrics['item_map']:.4f}")
    logger.info(f"  Item Coverage@10: {metrics['item_coverage@10']:.4f}")
    logger.info(f"  Item Diversity@10: {metrics['item_diversity@10']:.4f}")
    logger.info(f"  Item Novelty@10: {metrics['item_novelty@10']:.4f}")
    
    logger.info("\nTrack B: Stopping Quality")
    logger.info(f"  Stop Hit@10: {metrics['stop_hit@10']:.4f}")
    logger.info(f"  Stop NDCG@10: {metrics['stop_ndcg@10']:.4f}")
    logger.info(f"  Stop Rank (median): {metrics['stop_rank_median']}")
    logger.info(f"  Stop Rank (mean): {metrics['stop_rank_mean']:.2f}")
    logger.info(f"  Stop Prediction Rate: {metrics['stop_prediction_rate']:.4f}") 