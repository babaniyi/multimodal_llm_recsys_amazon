import json
import pandas as pd
from pathlib import Path
from ..baselines.recommender import CollaborativeFiltering, ContentBasedRecommender, HybridRecommender, PopularityRecommender, MatrixFactorization
from ..evaluation.metrics import calculate_metrics
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

# Initialize models
cf_model = CollaborativeFiltering(n_factors=num_items * 3)
content_model = ContentBasedRecommender()
hybrid_model = HybridRecommender(n_factors=num_items * 3)
popularity_model = PopularityRecommender()
mf_model = MatrixFactorization(n_factors=num_items * 3, learning_rate=0.01, n_epochs=20, reg=0.01)

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
    
    # Get predictions from each model
    cf_predictions = cf_model.predict(user_id, n_items=num_items)
    content_predictions = content_model.predict(item_id, n_items=num_items)
    hybrid_predictions = hybrid_model.predict(user_id, item_id, n_items=num_items)
    popularity_predictions = popularity_model.predict(n_items=num_items)
    mf_predictions = mf_model.predict(user_id, n_items=num_items)
    
    # Add predictions to entry
    entry['cf_model'] = cf_predictions
    entry['content_model'] = content_predictions
    entry['hybrid_model'] = hybrid_predictions
    entry['popularity_model'] = popularity_predictions
    entry['mf_model'] = mf_predictions

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

# Evaluate and compare models
models = ['gpt2-medium', 'cf_model', 'content_model', 'hybrid_model', 'popularity_model', 'mf_model']
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