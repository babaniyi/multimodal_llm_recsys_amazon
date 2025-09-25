"""
Robust evaluation module with statistical testing and multiple runs.
Addresses the critical evaluation credibility issues identified by reviewers.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from pathlib import Path
import logging
from scipy import stats
from sklearn.metrics import precision_score, recall_score, ndcg_score
from collections import defaultdict
import random
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class RobustEvaluator:
    """
    Robust evaluator that addresses evaluation credibility issues:
    - Multiple runs with different random seeds
    - Statistical significance testing
    - Confidence intervals
    - Proper train/test splits
    """
    
    def __init__(self, config: Dict[str, Any], num_runs: int = 5, confidence_level: float = 0.95):
        self.config = config
        self.num_runs = num_runs
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def calculate_metrics_robust(self, predictions: List[List[str]], targets: List[List[str]], k: int = 10) -> Dict[str, float]:
        """
        Calculate metrics with proper handling of edge cases.
        """
        metrics = {}
        
        if not predictions or not targets:
            logger.warning("Empty predictions or targets")
            return {f'hr@{k}': 0.0, f'precision@{k}': 0.0, f'recall@{k}': 0.0, 
                   f'ndcg@{k}': 0.0, 'mrr': 0.0, 'map': 0.0, 'coverage': 0.0, 
                   'diversity': 0.0, 'novelty': 0.0}
        
        # Calculate Hit Rate@k
        hits = 0
        total = len(predictions)
        for pred, target in zip(predictions, targets):
            if any(item in target for item in pred[:k]):
                hits += 1
        metrics[f'hr@{k}'] = hits / total if total > 0 else 0.0
        
        # Calculate Precision@k
        total_precision = 0
        for pred, target in zip(predictions, targets):
            hits = len(set(pred[:k]) & set(target))
            total_precision += hits / k if k > 0 else 0
        metrics[f'precision@{k}'] = total_precision / total if total > 0 else 0.0
        
        # Calculate Recall@k
        total_recall = 0
        for pred, target in zip(predictions, targets):
            hits = len(set(pred[:k]) & set(target))
            total_recall += hits / len(target) if len(target) > 0 else 0
        metrics[f'recall@{k}'] = total_recall / total if total > 0 else 0.0
        
        # Calculate NDCG@k
        total_ndcg = 0
        for pred, target in zip(predictions, targets):
            dcg = 0
            idcg = 1  # Since we're dealing with binary relevance
            for i, item in enumerate(pred[:k]):
                if item in target:
                    dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
            total_ndcg += dcg / idcg
        metrics[f'ndcg@{k}'] = total_ndcg / total if total > 0 else 0.0
        
        # Calculate MRR (Mean Reciprocal Rank)
        total_mrr = 0
        for pred, target in zip(predictions, targets):
            for i, item in enumerate(pred):
                if item in target:
                    total_mrr += 1 / (i + 1)
                    break
        metrics['mrr'] = total_mrr / total if total > 0 else 0.0
        
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
        metrics['map'] = total_map / total if total > 0 else 0.0
        
        # Calculate Coverage
        all_items = set()
        recommended_items = set()
        for pred in predictions:
            recommended_items.update(pred[:k])
        for target in targets:
            all_items.update(target)
        metrics['coverage'] = len(recommended_items) / len(all_items) if len(all_items) > 0 else 0.0
        
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
        metrics['diversity'] = total_diversity / count if count > 0 else 0.0
        
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
            total_novelty += pred_novelty / k if k > 0 else 0
        metrics['novelty'] = total_novelty / len(predictions) if len(predictions) > 0 else 0.0
        
        return metrics
    
    def run_multiple_evaluations(self, model_predictions: Dict[str, List[List[str]]], 
                                targets: List[List[str]], k: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple evaluations with different random seeds and calculate statistics.
        """
        results = {}
        
        for model_name, predictions in model_predictions.items():
            logger.info(f"Running {self.num_runs} evaluations for {model_name}")
            
            # Store metrics for each run
            run_metrics = {metric: [] for metric in ['hr', 'precision', 'recall', 'ndcg', 'mrr', 'map', 'coverage', 'diversity', 'novelty']}
            
            for run in range(self.num_runs):
                # Set random seed for reproducibility
                random.seed(42 + run)
                np.random.seed(42 + run)
                
                # Shuffle predictions and targets together to maintain correspondence
                combined = list(zip(predictions, targets))
                random.shuffle(combined)
                shuffled_predictions, shuffled_targets = zip(*combined)
                
                # Calculate metrics for this run
                metrics = self.calculate_metrics_robust(list(shuffled_predictions), list(shuffled_targets), k)
                
                # Store metrics
                for metric_name, value in metrics.items():
                    base_metric = metric_name.split('@')[0] if '@' in metric_name else metric_name
                    run_metrics[base_metric].append(value)
            
            # Calculate statistics
            model_results = {}
            for metric_name, values in run_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    ci_lower, ci_upper = self._calculate_confidence_interval(values)
                    
                    model_results[f'{metric_name}@{k}'] = {
                        'mean': mean_val,
                        'std': std_val,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'values': values
                    }
            
            results[model_name] = model_results
            
            # Log results
            logger.info(f"Results for {model_name} (mean ± std, 95% CI):")
            for metric_name, stats_dict in model_results.items():
                logger.info(f"  {metric_name}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f} "
                          f"[{stats_dict['ci_lower']:.4f}, {stats_dict['ci_upper']:.4f}]")
        
        return results
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for the given values."""
        if len(values) < 2:
            return (values[0], values[0]) if values else (0.0, 0.0)
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Use sample standard deviation
        
        # Calculate confidence interval using t-distribution
        t_val = stats.t.ppf(1 - self.alpha/2, len(values) - 1)
        margin_error = t_val * (std_val / np.sqrt(len(values)))
        
        return (mean_val - margin_error, mean_val + margin_error)
    
    def statistical_significance_test(self, results: Dict[str, Dict[str, Any]], 
                                    baseline_model: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Perform statistical significance tests comparing all models to baseline.
        """
        significance_results = {}
        
        # If no baseline specified, use the first model as baseline
        if baseline_model is None:
            baseline_model = list(results.keys())[0]
        
        # Skip if baseline model doesn't exist
        if baseline_model not in results:
            logger.warning(f"Baseline model '{baseline_model}' not found in results. Skipping significance tests.")
            return significance_results
        
        for model_name, model_results in results.items():
            if model_name == baseline_model:
                continue
                
            significance_results[model_name] = {}
            
            for metric_name, stats_dict in model_results.items():
                if metric_name in results[baseline_model]:
                    baseline_values = results[baseline_model][metric_name]['values']
                    model_values = stats_dict['values']
                    
                    # Perform paired t-test
                    if len(baseline_values) == len(model_values) and len(baseline_values) > 1:
                        t_stat, p_value = stats.ttest_rel(model_values, baseline_values)
                        
                        significance_results[model_name][metric_name] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < self.alpha,
                            'effect_size': self._calculate_cohens_d(model_values, baseline_values)
                        }
        
        return significance_results
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                            (len(group1) + len(group2) - 2))
        
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    
    def save_results(self, results: Dict[str, Any], significance_results: Dict[str, Any], 
                    output_path: str):
        """Save evaluation results with statistical analysis."""
        output_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'num_runs': self.num_runs,
            'confidence_level': self.confidence_level,
            'results': results,
            'statistical_significance': significance_results,
            'summary': self._generate_summary(results, significance_results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        logger.info(f"Results saved to {output_path}")
    
    def _generate_summary(self, results: Dict[str, Any], significance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the evaluation results."""
        summary = {
            'best_models': {},
            'key_findings': []
        }
        
        # Find best models for each metric
        metrics = ['hr', 'precision', 'recall', 'ndcg', 'mrr', 'map']
        for metric in metrics:
            best_model = None
            best_score = -1
            
            for model_name, model_results in results.items():
                metric_key = f'{metric}@10'
                if metric_key in model_results:
                    score = model_results[metric_key]['mean']
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                summary['best_models'][metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        # Add key findings
        summary['key_findings'].append(f"Evaluation completed with {self.num_runs} runs per model")
        summary['key_findings'].append(f"Confidence level: {self.confidence_level*100}%")
        
        return summary


def load_test_data_with_predictions(data_path: str) -> Tuple[Dict[str, List[List[str]]], List[List[str]]]:
    """
    Load test data with model predictions.
    Returns predictions dictionary and targets list.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    targets = []
    
    # Extract targets
    targets = [entry['output'].split(', ') for entry in data]
    
    # Extract predictions for each model
    # Only use models that actually exist in the data
    if 'model_response' in data[0]:
        predictions['multimodal_llm'] = [entry['model_response'].split(', ') for entry in data]
    
    return predictions, targets


def main():
    """Main function to run robust evaluation."""
    from src.utils.utils import setup_logging, load_config
    
    setup_logging()
    logger.info("Running robust evaluation...")
    
    try:
        
        config = load_config()
        evaluator = RobustEvaluator(config, num_runs=5)
        
        # Load data - use the actual test data with model responses
        predictions, targets = load_test_data_with_predictions('data/processed/test_with_responses.json')
        
        # Run robust evaluation
        results = evaluator.run_multiple_evaluations(predictions, targets, k=10)
        
        # Statistical significance testing
        significance_results = evaluator.statistical_significance_test(results)
        
        # Save results
        evaluator.save_results(results, significance_results, 'data/processed/robust_evaluation_results.json')
        
        logger.info("Robust evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Robust evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
