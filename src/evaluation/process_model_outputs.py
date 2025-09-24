import json
from src.utils.utils import load_config
from src.evaluation.ablation_studies import AblationStudy
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load config
config = load_config()
num_items = config['data_config']['number_of_items_to_predict']

# Check if ablation studies are enabled
RUN_ABLATION_STUDIES = config['training_config'].get('ablation_studies', False)

# Load the JSON data
with open('data/processed/test_with_responses.json', 'r') as file:
    data = json.load(file)

# Process each entry
for entry in data:
    # Split the model_response into items
    model_response_items = entry['model_response'].split(', ')
    
    # Truncate or pad the model_response to num_items
    if len(model_response_items) < num_items:
        model_response_items.extend(['<|endoftext|>'] * (num_items - len(model_response_items)))
    else:
        model_response_items = model_response_items[:num_items]
    
    # Join the items back into a string
    entry['model_response'] = ', '.join(model_response_items)

# Save the modified data to a new JSON file for evaluation
with open('data/processed/test_with_responses_processed.json', 'w') as file:
    json.dump(data, file, indent=4)

logger.info("Model outputs processed successfully!")

# Run ablation studies if enabled
if RUN_ABLATION_STUDIES:
    logger.info("Running ablation studies...")
    try:
        # Load training and test data for ablation studies
        with open('data/processed/train.json', 'r') as f:
            train_data = json.load(f)
        
        with open('data/processed/test.json', 'r') as f:
            test_data = json.load(f)
        
        # Initialize ablation study
        ablation_study = AblationStudy(config)
        
        # Run ablation studies
        ablation_results = ablation_study.run_complete_ablation(train_data, test_data, {})
        
        logger.info("Ablation studies completed successfully!")
        logger.info(f"Best modality: {ablation_results['summary']['best_modality']}")
        logger.info(f"Best fusion method: {ablation_results['summary']['best_fusion_method']}")
        
    except Exception as e:
        logger.error(f"Ablation studies failed: {e}")
        logger.info("Continuing without ablation studies...")
else:
    logger.info("Ablation studies disabled. Set 'ablation_studies': true in config to enable.")


def main():
    """Main function to process model outputs."""
    logger.info("Processing model outputs...")
    
    # The processing logic is already executed when the module is imported
    # This function exists for compatibility with the workflow script
    
    logger.info("Model output processing completed!")


if __name__ == "__main__":
    main() 