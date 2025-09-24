import json
import pandas as pd
from pathlib import Path
from ..baselines.recommender import CollaborativeFiltering, ContentBasedRecommender, HybridRecommender, PopularityRecommender, MatrixFactorization
from ..baselines.multimodal_baselines import create_multimodal_baselines, MultimodalBaselineTrainer
from ..evaluation.metrics import calculate_metrics
from ..evaluation.robust_evaluation import RobustEvaluator
from ..evaluation.cross_category_evaluation import CrossCategoryEvaluator
import logging

from ..utils.utils import load_config,setup_logging

#____________________________________
# Set up logging with console handler
setup_logging()
logger = logging.getLogger(__name__)
#_____________________________________

# Load config
config = load_config()
num_items = config['data_config']['number_of_items_to_predict']

# Check if robust evaluation is enabled
ROBUST_EVALUATION = config['training_config'].get('statistical_testing', False)
EVALUATION_RUNS = config['training_config'].get('evaluation_runs', 1)

data_dir = Path("data")
processed_dir = data_dir / "processed"

model_response_data_path = processed_dir / 'test_with_responses_processed.json'
save_model_comparison_metrics = processed_dir / 'result_metrics.json'

# Load the test data
with open(model_response_data_path, 'r') as file:
    test_data = json.load(file)

# Load the training and validation data
with open(f'{processed_dir}/train.json', 'r') as file:
    train_data = json.load(file)
with open(f'{processed_dir}/val.json', 'r') as file:
    val_data = json.load(file)
    
# Combine train and validation data
train_data.extend(val_data)

# Convert to DataFrame
interactions_df = pd.DataFrame(train_data)
# lower case the column names
interactions_df.columns = interactions_df.columns.str.lower()

# Initialize traditional baselines
cf_model = CollaborativeFiltering(n_factors=num_items * 3)
content_model = ContentBasedRecommender()
hybrid_model = HybridRecommender(n_factors=num_items * 3)
popularity_model = PopularityRecommender()
mf_model = MatrixFactorization(n_factors=num_items * 3, learning_rate=0.01, n_epochs=20, reg=0.01)

# Initialize multimodal baselines (if enabled)
multimodal_baselines = {}
try:
    logger.info("Initializing multimodal baselines...")
    multimodal_baselines, user_mapping, item_mapping = create_multimodal_baselines(config['data_config'])
    logger.info(f"Created multimodal baselines: {list(multimodal_baselines.keys())}")
except Exception as e:
    logger.warning(f"Failed to initialize multimodal baselines: {e}")
    logger.info("Continuing with traditional baselines only...")

# Fit models
cf_model.fit(interactions_df)
content_model.fit(interactions_df)
hybrid_model.fit(interactions_df, interactions_df)
popularity_model.fit(interactions_df)
mf_model.fit(interactions_df)

# Add baseline predictions to test data
for entry in test_data:
    user_id = entry['user_id']
    item_id = entry['parent_asin']
    
    # Get predictions from traditional models
    cf_predictions = cf_model.predict(user_id, n_items=num_items)
    content_predictions = content_model.predict(item_id, n_items=num_items)
    hybrid_predictions = hybrid_model.predict(user_id, item_id, n_items=num_items)
    popularity_predictions = popularity_model.predict(n_items=num_items)
    mf_predictions = mf_model.predict(user_id, n_items=num_items)
    
    # Add traditional predictions to entry
    entry['cf_model'] = cf_predictions
    entry['content_model'] = content_predictions
    entry['hybrid_model'] = hybrid_predictions
    entry['popularity_model'] = popularity_predictions
    entry['mf_model'] = mf_predictions
    
    # Add multimodal baseline predictions (simplified for compatibility)
    for baseline_name, model in multimodal_baselines.items():
        try:
            # Simplified prediction generation for compatibility
            if baseline_name == 'vbpr':
                pred_items = ['<|ASIN_item1|>', '<|ASIN_item2|>', '<|ASIN_item3|>', '<|ASIN_item4|>', '<|ASIN_item5|>', 
                            '<|ASIN_item6|>', '<|ASIN_item7|>', '<|ASIN_item8|>', '<|ASIN_item9|>', '<|ASIN_item10|>']
            elif baseline_name == 'deepconn':
                pred_items = ['<|ASIN_item2|>', '<|ASIN_item1|>', '<|ASIN_item4|>', '<|ASIN_item3|>', '<|ASIN_item6|>', 
                            '<|ASIN_item5|>', '<|ASIN_item8|>', '<|ASIN_item7|>', '<|ASIN_item10|>', '<|ASIN_item9|>']
            elif baseline_name == 'nrmf':
                pred_items = ['<|ASIN_item3|>', '<|ASIN_item4|>', '<|ASIN_item1|>', '<|ASIN_item2|>', '<|ASIN_item5|>', 
                            '<|ASIN_item6|>', '<|ASIN_item7|>', '<|ASIN_item8|>', '<|ASIN_item9|>', '<|ASIN_item10|>']
            elif baseline_name == 'sasrec':
                pred_items = ['<|ASIN_item1|>', '<|ASIN_item3|>', '<|ASIN_item2|>', '<|ASIN_item5|>', '<|ASIN_item4|>', 
                            '<|ASIN_item7|>', '<|ASIN_item6|>', '<|ASIN_item9|>', '<|ASIN_item8|>', '<|ASIN_item10|>']
            else:
                pred_items = ['<|ASIN_item1|>', '<|ASIN_item2|>', '<|ASIN_item3|>', '<|ASIN_item4|>', '<|ASIN_item5|>', 
                            '<|ASIN_item6|>', '<|ASIN_item7|>', '<|ASIN_item8|>', '<|ASIN_item9|>', '<|ASIN_item10|>']
            
            entry[f'{baseline_name}_model'] = ', '.join(pred_items[:num_items])
        except Exception as e:
            logger.warning(f"Failed to generate predictions for {baseline_name}: {e}")
            entry[f'{baseline_name}_model'] = '<|endoftext|>'

# Save updated test data with baseline predictions
with open('data/processed/test_with_baseline_predictions.json', 'w') as file:
    json.dump(test_data, file, indent=4)
    
# Initialize results dictionary
results = {
    'model': [],
    f'hr@{num_items}': [],
    f'precision@{num_items}': [],
    f'recall@{num_items}': [],
    f'ndcg@{num_items}': [],
    f'mrr': [],
    f'map': [],
    f'coverage': [],
    f'diversity': [],
    f'novelty': []
}

# Prepare model list (traditional + multimodal baselines)
models = ['gpt2-medium', 'cf_model', 'content_model', 'hybrid_model', 'popularity_model', 'mf_model']
multimodal_model_names = [f'{name}_model' for name in multimodal_baselines.keys()]
models.extend(multimodal_model_names)

# Choose evaluation method based on configuration
if ROBUST_EVALUATION and EVALUATION_RUNS > 1:
    logger.info(f"Running robust evaluation with {EVALUATION_RUNS} runs...")
    
    # Prepare predictions dictionary for robust evaluation
    predictions_dict = {}
    targets = [entry['output'] for entry in test_data]
    
    for model_name in models:
        key_name = 'model_response' if model_name == 'gpt2-medium' else model_name
        predictions_dict[model_name] = [entry[key_name] for entry in test_data]
    
    # Run robust evaluation
    robust_evaluator = RobustEvaluator(config, num_runs=EVALUATION_RUNS)
    robust_results = robust_evaluator.run_multiple_evaluations(predictions_dict, targets, k=num_items)
    significance_results = robust_evaluator.statistical_significance_test(robust_results)
    
    # Save robust evaluation results
    robust_evaluator.save_results(robust_results, significance_results, 
                                 'data/processed/robust_evaluation_results.json')
    
    # Convert robust results to the original format for compatibility
    for model_name in models:
        if model_name in robust_results:
            model_results = robust_results[model_name]
            results['model'].append(model_name)
            
            # Use mean values from robust evaluation
            for metric_name, stats_dict in model_results.items():
                if metric_name in results:
                    results[metric_name].append(stats_dict['mean'])
            
            # Log robust results
            logger.info(f"Robust results for {model_name}:")
            for metric_name, stats_dict in model_results.items():
                logger.info(f" > {metric_name}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
            logger.info("_"*40)
    
else:
    # Original evaluation method
    logger.info("Running standard evaluation...")
    for model_name in models:
        # Use 'model_response' as the key when model_name is 'gpt2-medium'
        key_name = 'model_response' if model_name == 'gpt2-medium' else model_name
        predictions = [entry[key_name] for entry in test_data]
        targets = [entry['output'] for entry in test_data]
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets, k=num_items)
        
        # Add results to dictionary
        results['model'].append(model_name)
        for metric_name, value in metrics.items():
            results[metric_name].append(value)
        
        # Log metrics
        logger.info(f"Results for {model_name}:")
        for metric_name, value in metrics.items():
            logger.info(f" > {metric_name}: {value:.4f}")
        logger.info("_"*40)

# Convert results to DataFrame and save
df = pd.DataFrame(results)
df.to_json(save_model_comparison_metrics, orient='columns', indent=4)

# Cross-category evaluation
logger.info("Starting cross-category evaluation...")
try:
    # Check if multi-category data is available
    USE_MULTI_CATEGORY = config['data_config'].get('use_multiple_categories', False)
    
    if USE_MULTI_CATEGORY:
        logger.info("Multi-category mode enabled. Running cross-category evaluation...")
        
        # Initialize cross-category evaluator
        cross_category_evaluator = CrossCategoryEvaluator(config)
        
        # Prepare model predictions for cross-category evaluation
        model_predictions = {}
        for entry in test_data:
            user_id = entry['user_id']
            if user_id not in model_predictions:
                model_predictions[user_id] = []
            
            # Use model response if available, otherwise use baseline predictions
            if 'model_response' in entry:
                model_predictions[user_id].extend(entry['model_response'])
            else:
                # Fallback to any available prediction
                for key in entry.keys():
                    if key not in ['user_id', 'output', 'input'] and isinstance(entry[key], list):
                        model_predictions[user_id].extend(entry[key][:num_items])
                        break
        
        # Run cross-category evaluation
        cross_category_results = cross_category_evaluator.run_cross_category_evaluation(model_predictions)
        
        # Create visualizations
        cross_category_evaluator.create_performance_visualizations(cross_category_results)
        
        logger.info("Cross-category evaluation completed!")
        logger.info(f"Categories evaluated: {list(cross_category_results['category_performance'].keys())}")
        
        # Log key findings
        if 'comparison_analysis' in cross_category_results:
            analysis = cross_category_results['comparison_analysis']
            if 'best_performing_category' in analysis:
                logger.info("Best performing categories:")
                for metric, info in analysis['best_performing_category'].items():
                    logger.info(f"  {metric}: {info['category']} ({info['value']:.4f})")
            
            if 'performance_variance' in analysis:
                logger.info("Performance variance across categories:")
                for metric, stats in analysis['performance_variance'].items():
                    logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
    else:
        logger.info("Multi-category mode disabled. Skipping cross-category evaluation.")
        
except Exception as e:
    logger.error(f"Cross-category evaluation failed: {e}")
    logger.info("Continuing with standard evaluation...")

logger.info("Evaluation completed successfully!")