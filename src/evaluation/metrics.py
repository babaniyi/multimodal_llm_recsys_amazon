import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.metrics import precision_score, recall_score, ndcg_score
from collections import defaultdict

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
    
    # Calculate Hit Rate@k
    hits = 0
    total = len(predictions)
    for pred, target in zip(predictions, targets):
        if any(item in target for item in pred[:k]):
            hits += 1
    metrics[f'hr@{k}'] = hits / total
    
    # Calculate Precision@k
    total_precision = 0
    for pred, target in zip(predictions, targets):
        hits = len(set(pred[:k]) & set(target))
        total_precision += hits / k
    metrics[f'precision@{k}'] = total_precision / total
    
    # Calculate Recall@k
    total_recall = 0
    for pred, target in zip(predictions, targets):
        hits = len(set(pred[:k]) & set(target))
        total_recall += hits / len(target)
    metrics[f'recall@{k}'] = total_recall / total
    
    # Calculate NDCG@k
    total_ndcg = 0
    for pred, target in zip(predictions, targets):
        dcg = 0
        idcg = 1  # Since we're dealing with binary relevance
        for i, item in enumerate(pred[:k]):
            if item in target:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
        total_ndcg += dcg / idcg
    metrics[f'ndcg@{k}'] = total_ndcg / total
    
    # Calculate MRR (Mean Reciprocal Rank)
    total_mrr = 0
    for pred, target in zip(predictions, targets):
        for i, item in enumerate(pred):
            if item in target:
                total_mrr += 1 / (i + 1)
                break
    metrics['mrr'] = total_mrr / total
    
    # Calculate MAP (Mean Average Precision)
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

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate the model on the test set.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        k: Number of items to consider for top-k metrics
        
    Returns:
        Dictionary of metric names and their values
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_embeddings = batch.get("image_embeddings", None)
            if image_embeddings is not None:
                image_embeddings = image_embeddings.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeddings=image_embeddings
            )
            
            # Get predictions
            pred_ids = torch.argmax(outputs, dim=-1)
            
            # Convert predictions and targets to text
            for i in range(len(pred_ids)):
                pred_text = model.tokenizer.decode(pred_ids[i])
                target_text = model.tokenizer.decode(labels[i])
                all_predictions.append(pred_text)
                all_targets.append(target_text)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets, k)
    
    # Log metrics
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    return metrics 