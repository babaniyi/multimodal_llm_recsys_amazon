import json
from ..utils.utils import load_config
from pathlib import Path

# Load config
config = load_config()
num_items = config['data_config']['number_of_items_to_predict']

data_dir = Path("data")
processed_dir = data_dir / "processed"

model_response_data_path = processed_dir / 'test_with_responses.json'
save_new_response_data_path = processed_dir / 'test_with_responses_processed.json'


# Load the JSON data
with open(model_response_data_path, 'r') as file:
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
with open(save_new_response_data_path, 'w') as file:
    json.dump(data, file, indent=4) 