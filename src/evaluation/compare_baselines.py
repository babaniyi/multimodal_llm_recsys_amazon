import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json
import torch
from tqdm import tqdm
from scipy import stats
from itertools import combinations

from ..baselines.recommender import (
    CollaborativeFiltering,
    ContentBasedRecommender,
    HybridRecommender,
    PopularityRecommender
)
from ..evaluation.metrics import calculate_metrics
from ..models.multimodal_gpt import MultiModalGPT
from ..data.loader import create_data_loaders

logger = logging.getLogger(__name__)

def prepare_data_for_baselines(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for baseline models.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        
    Returns:
        Tuple of (interactions_df, items_df, image_embeddings)
    """
    # Create interactions DataFrame
    interactions = []
    for data in [train_data, val_data, test_data]:
        for item in data:
            interactions.append({
                'user_id': item['user_id'],
                'item_id': item['item_id'],
                'rating': item.get('rating', 1.0)  # Default rating if not present
            })
    interactions_df = pd.DataFrame(interactions)
    
    # Create items DataFrame
    items = {}
    for data in [train_data, val_data, test_data]:
        for item in data:
            if item['item_id'] not in items:
                items[item['item_id']] = {
                    'item_id': item['item_id'],
                    'title': item.get('title', ''),
                    'description': item.get('description', '')
                }
    items_df = pd.DataFrame(list(items.values()))
    
    # Extract image embeddings if available
    image_embeddings = {}
    for data in [train_data, val_data, test_data]:
        for item in data:
            if 'image_embeddings' in item and item['item_id'] not in image_embeddings:
                image_embeddings[item['item_id']] = item['image_embeddings']
    
    return interactions_df, items_df, image_embeddings

def evaluate_baselines(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    k: int = 10
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[List[int]]], List[List[int]]]:
    """
    Evaluate baseline recommendation methods.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        k: Number of items to consider for top-k metrics
        
    Returns:
        Tuple of (results dictionary, predictions dictionary, ground truth)
    """
    # Prepare data
    interactions_df, items_df, image_embeddings = prepare_data_for_baselines(
        train_data, val_data, test_data
    )
    
    # Initialize models
    cf_model = CollaborativeFiltering(n_factors=100)
    content_model = ContentBasedRecommender(image_embeddings)
    hybrid_model = HybridRecommender(cf_weight=0.5, content_weight=0.5)
    popularity_model = PopularityRecommender()
    
    # Train models
    logger.info("Training Collaborative Filtering model...")
    cf_model.fit(interactions_df)
    
    logger.info("Training Content-Based model...")
    content_model.fit(items_df)
    
    logger.info("Training Hybrid model...")
    hybrid_model.fit(interactions_df, items_df, image_embeddings)
    
    logger.info("Training Popularity-Based model...")
    popularity_model.fit(interactions_df)
    
    # Generate predictions
    results = {}
    predictions = {}
    
    # Get ground truth next items from test data
    ground_truth = [item['next_items'] for item in test_data]
    
    # Collaborative Filtering predictions
    logger.info("Generating Collaborative Filtering predictions...")
    cf_predictions = []
    for item in tqdm(test_data):
        pred_items = cf_model.predict(item['user_id'], k)
        cf_predictions.append(pred_items)
    
    # Content-Based predictions
    logger.info("Generating Content-Based predictions...")
    content_predictions = []
    for item in tqdm(test_data):
        pred_items = content_model.predict(item['item_id'], k)
        content_predictions.append(pred_items)
    
    # Hybrid predictions
    logger.info("Generating Hybrid predictions...")
    hybrid_predictions = []
    for item in tqdm(test_data):
        pred_items = hybrid_model.predict(item['user_id'], item['item_id'], k)
        hybrid_predictions.append(pred_items)
    
    # Popularity-Based predictions
    logger.info("Generating Popularity-Based predictions...")
    popular_items = popularity_model.predict(k)
    popularity_predictions = [popular_items for _ in range(len(test_data))]
    
    # Store predictions
    predictions['collaborative_filtering'] = cf_predictions
    predictions['content_based'] = content_predictions
    predictions['hybrid'] = hybrid_predictions
    predictions['popularity_based'] = popularity_predictions
    
    # Calculate metrics
    results['collaborative_filtering'] = calculate_metrics(
        cf_predictions,
        ground_truth,
        k
    )
    
    results['content_based'] = calculate_metrics(
        content_predictions,
        ground_truth,
        k
    )
    
    results['hybrid'] = calculate_metrics(
        hybrid_predictions,
        ground_truth,
        k
    )
    
    results['popularity_based'] = calculate_metrics(
        popularity_predictions,
        ground_truth,
        k
    )
    
    return results, predictions, ground_truth

def compare_with_multimodal(
    multimodal_model: MultiModalGPT,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    baseline_results: Dict[str, Dict[str, float]],
    k: int = 10
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[List[int]]], List[List[int]]]:
    """
    Compare multimodal model with baselines.
    
    Args:
        multimodal_model: Trained multimodal model
        test_loader: Test data loader
        device: Device to run evaluation on
        baseline_results: Results from baseline models
        k: Number of items to consider for top-k metrics
        
    Returns:
        Tuple of (results dictionary, predictions dictionary, ground truth)
    """
    # Get multimodal model predictions
    multimodal_model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating multimodal predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_embeddings = batch.get("image_embeddings", None)
            if image_embeddings is not None:
                image_embeddings = image_embeddings.to(device)
            
            outputs = multimodal_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeddings=image_embeddings
            )
            
            # Get top-k predictions
            logits = outputs.logits[:, -1, :]
            pred_ids = torch.topk(logits, k=k)[1].cpu().numpy()
            
            # Store predictions and ground truth
            predictions.extend(pred_ids.tolist())
            ground_truth.extend(batch['next_items'].tolist())
    
    # Calculate multimodal metrics
    multimodal_results = calculate_metrics(
        predictions,
        ground_truth,
        k
    )
    
    # Combine all results
    all_results = {
        'multimodal_llm': multimodal_results,
        **baseline_results
    }
    
    # Create predictions dictionary
    all_predictions = {
        'multimodal_llm': predictions,
        **baseline_results['predictions']
    }
    
    return all_results, all_predictions, ground_truth

def plot_comparison_results(results: Dict[str, Dict[str, float]], output_dir: Path):
    """
    Plot comparison results for all models.
    
    Args:
        results: Dictionary of metrics for all models
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn')
    
    # 1. Performance Metrics Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = ['hr@10', 'precision@10', 'recall@10', 'ndcg@10']
    titles = ['Hit Rate@10', 'Precision@10', 'Recall@10', 'NDCG@10']
    
    for (metric, title), ax in zip(zip(metrics, titles), axes.flat):
        values = [results[model][metric] for model in results.keys()]
        models = list(results.keys())
        
        # Create bar plot
        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(title)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png')
    plt.close()
    
    # 2. Additional Metrics Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Additional Model Metrics', fontsize=16)
    
    metrics = ['mrr', 'map', 'coverage', 'diversity']
    titles = ['Mean Reciprocal Rank', 'Mean Average Precision', 'Coverage', 'Diversity']
    
    for (metric, title), ax in zip(zip(metrics, titles), axes.flat):
        values = [results[model][metric] for model in results.keys()]
        models = list(results.keys())
        
        # Create bar plot
        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(title)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'additional_metrics.png')
    plt.close()
    
    # 3. Radar Chart for Overall Performance
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Select metrics for radar chart
    radar_metrics = ['hr@10', 'precision@10', 'recall@10', 'ndcg@10', 'mrr', 'map']
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False)
    
    # Close the plot by appending the first value
    angles = np.concatenate((angles, [angles[0]]))
    
    # Plot for each model
    for model in results.keys():
        values = [results[model][metric] for metric in radar_metrics]
        values = np.concatenate((values, [values[0]]))  # Close the plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)
    ax.set_ylim(0, 1)
    plt.title('Overall Model Performance (Radar Chart)')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png')
    plt.close()
    
    # 4. Heatmap of Model Similarities
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate similarity matrix using NDCG@10
    models = list(results.keys())
    similarity_matrix = np.zeros((len(models), len(models)))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            similarity_matrix[i, j] = 1 - abs(results[model1]['ndcg@10'] - results[model2]['ndcg@10'])
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        xticklabels=models,
        yticklabels=models,
        cmap='YlOrRd',
        ax=ax
    )
    plt.title('Model Similarity Heatmap (based on NDCG@10)')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_similarity.png')
    plt.close()

def perform_statistical_tests(
    results: Dict[str, Dict[str, float]],
    predictions: Dict[str, List[List[int]]],
    ground_truth: List[List[int]],
    k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Perform statistical significance tests between models.
    
    Args:
        results: Dictionary of metrics for all models
        predictions: Dictionary of predictions for each model
        ground_truth: Ground truth next items
        k: Number of items to consider for top-k metrics
        
    Returns:
        Dictionary of p-values for each metric and model pair
    """
    statistical_tests = {}
    
    # Get all model pairs
    model_pairs = list(combinations(results.keys(), 2))
    
    # Metrics to test
    metrics = ['hr@10', 'precision@10', 'recall@10', 'ndcg@10']
    
    for metric in metrics:
        statistical_tests[metric] = {}
        for model1, model2 in model_pairs:
            # Calculate metric values for each prediction
            model1_scores = []
            model2_scores = []
            
            for pred1, pred2, gt in zip(predictions[model1], predictions[model2], ground_truth):
                # Calculate metric for each prediction
                score1 = calculate_metrics([pred1], [gt], k)[metric]
                score2 = calculate_metrics([pred2], [gt], k)[metric]
                model1_scores.append(score1)
                model2_scores.append(score2)
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
            statistical_tests[metric][f"{model1}_vs_{model2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return statistical_tests

def main():
    # Load data
    with open("data/processed/train.json", "r") as f:
        train_data = json.load(f)
    with open("data/processed/val.json", "r") as f:
        val_data = json.load(f)
    with open("data/processed/test.json", "r") as f:
        test_data = json.load(f)
    
    # Load multimodal model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("configs/train_config.json", "r") as f:
        config = json.load(f)
    
    multimodal_model = MultiModalGPT(config["model_config"])
    multimodal_model.load_state_dict(torch.load("checkpoints/best_model.pt"))
    multimodal_model.to(device)
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_length=config["max_length"]
    )
    
    # Evaluate baselines
    logger.info("Evaluating baseline models...")
    baseline_results, baseline_predictions, ground_truth = evaluate_baselines(train_data, val_data, test_data)
    
    # Compare with multimodal model
    logger.info("Comparing with multimodal model...")
    all_results, all_predictions, ground_truth = compare_with_multimodal(
        multimodal_model,
        data_loaders["test"],
        device,
        {'results': baseline_results, 'predictions': baseline_predictions}
    )
    
    # Perform statistical tests
    logger.info("Performing statistical significance tests...")
    statistical_tests = perform_statistical_tests(
        all_results,
        all_predictions,
        ground_truth
    )
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump({
            'metrics': all_results,
            'statistical_tests': statistical_tests
        }, f, indent=4)
    
    # Plot results
    plot_comparison_results(all_results, output_dir)
    
    # Print results
    logger.info("\nModel Comparison Results:")
    for model_name, metrics in all_results.items():
        logger.info(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
    
    logger.info("\nStatistical Significance Tests:")
    for metric, tests in statistical_tests.items():
        logger.info(f"\n{metric.upper()}:")
        for test_name, test_results in tests.items():
            logger.info(f"{test_name}:")
            logger.info(f"  t-statistic: {test_results['t_statistic']:.4f}")
            logger.info(f"  p-value: {test_results['p_value']:.4f}")
            logger.info(f"  significant: {test_results['significant']}")

if __name__ == "__main__":
    main() 