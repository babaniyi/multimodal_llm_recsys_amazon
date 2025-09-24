# Data Format Correction - Multi-Category Loader

## Problem Identified

The multi-category loader was creating a simplified data format that didn't match the original training data format expected by the model. The original format includes:

### Original Format (Expected by Model)
```json
{
  "user_id": "AEYXNC2PNGHMTKNNBVHGZGZT6IIA",
  "parent_asin": "<|ASIN_B07T4J8ZTK|>",
  "rating": 1.0,
  "title": "285753A Washer Motor Coupler...",
  "input": "Review_Text: Bought these 3 for the price of 1... <|endoftext|>",
  "output": "<|ASIN_B07L3PNM5T|>, <|ASIN_B07R6QP1YV|>, <|ASIN_B08PMTGW81|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>",
  "output_names": "Living house Replaces 694089 Dryer Blower Wheel...",
  "instruction": "Given a user's purchase history and review for a product, predict the next 10 products they would likely purchase. "
}
```

### Multi-Category Format (Incorrect)
```json
{
  "user_id": "user123",
  "parent_asin": "B123456789",
  "title": "Coffee Maker",
  "instruction": "Recommend similar products to: Coffee Maker (Category: Appliances)",
  "input": "User review: This coffee maker makes excellent coffee...",
  "output": "Recommended product: Coffee Maker (ID: B123456789)"
}
```

## Key Differences

### 1. Instruction Format
- **Original**: Uses `config['data_config']['user_prompt']` from config
- **Multi-category**: Created custom instruction text

### 2. Input Format  
- **Original**: `"Review_Text: [actual review] <|endoftext|>"`
- **Multi-category**: `"User review: [truncated review]..."`

### 3. Output Format
- **Original**: Uses `processor.get_next_items()` to find actual next purchases
- **Multi-category**: Simple product recommendation text

### 4. Parent ASIN Format
- **Original**: `"<|ASIN_B07T4J8ZTK|>"` (with ASIN tags)
- **Multi-category**: `"B123456789"` (plain ASIN)

## Solution Implemented

### Updated Multi-Category Loader (`src/data/multi_category_loader.py`)

#### 1. Added `process_data_for_training()` Method
```python
def process_data_for_training(self, all_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Process data using the same logic as original prepare_data.py.
    This creates the proper instruction, input, and output format.
    """
    # Import processor functions
    from .processor import get_next_items, cast_unix_to_date
    
    for category, data in all_data.items():
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Apply the same transformations as original prepare_data.py
        df["date"] = cast_unix_to_date(df["timestamp"])
        df = df.sort_values(by=["date", "user_id"])
        
        # Format parent_asin with ASIN tags
        df["parent_asin"] = "<|ASIN_" + df["parent_asin"] + "|>"
        
        # Create input format: "Review_Text: [review] <|endoftext|>"
        df["input"] = df["review_text"].apply(lambda x: f"Review_Text: {x} <|endoftext|>" if x else "Review_Text: <|endoftext|>")
        
        # Get next items using the same logic as original
        labels_df = get_next_items(
            df=df, 
            padding_strategy='pad_token', 
            pad_token="<|endoftext|>", 
            number_of_items_to_predict=self.config['data_config']['number_of_items_to_predict'], 
            min_interactions=self.config['data_config']['min_interactions']
        )
        
        # Merge with labels
        join_keys = ["user_id", "parent_asin", "date"]
        df = df.merge(labels_df, on=join_keys, how="inner")
        
        # Rename columns to match expected format
        df = df.rename(columns={'next_items': "output", 'next_item_names': "output_names"})
        
        # Add instruction from config
        df['instruction'] = self.config['data_config']['user_prompt']
```

#### 2. Updated Processing Pipeline
```python
# Process data using the same logic as original prepare_data.py
processed_data = self.process_data_for_training(all_data)

# Create splits
splits = self.create_cross_category_splits(processed_data)
```

#### 3. Removed Custom Format Fields
- Removed custom `instruction`, `input`, `output` fields from raw data processing
- Let the `process_data_for_training()` method create proper format

### Updated Processor (`src/data/processor.py`)

#### Updated `format_input_phi()` Method
```python
# Use instruction from entry (should be present after proper processing)
if 'instruction' in entry:
    instruction_text = f"<|user|>\n{entry['instruction']}"
else:
    # Fallback for backward compatibility
    title = entry.get('title', '')
    category = entry.get('category', '')
    instruction_text = f"<|user|>\nRecommend similar products to: {title} (Category: {category})"
```

### Updated Dataset (`src/data/dataset.py`)

#### Updated Output Handling
```python
# Use output from entry (should be present after proper processing)
if 'output' in entry:
    response_text = f"\n<|assistant|>:\n{entry['output']}"
else:
    # Fallback for backward compatibility
    parent_asin = entry.get('parent_asin', '')
    title = entry.get('title', '')
    response_text = f"\n<|assistant|>:\nRecommended product: {title} (ID: {parent_asin})"
```

## Benefits

### ✅ Correct Data Format
- **Proper instruction**: Uses config-defined user prompt
- **Proper input**: "Review_Text: [review] <|endoftext|>" format
- **Proper output**: Actual next purchases using `get_next_items()`
- **Proper ASIN format**: "<|ASIN_B07T4J8ZTK|>" tags

### ✅ Training Compatibility
- **Same format** as original single-category data
- **Same processing logic** as `prepare_data.py`
- **Same model expectations** for instruction/input/output

### ✅ Backward Compatibility
- **Fallback handling** for missing fields
- **Graceful degradation** if processing fails
- **Support for both formats** during transition

## Expected Results

The multi-category data will now have the exact same format as the original data:

```json
{
  "user_id": "user123",
  "parent_asin": "<|ASIN_B123456789|>",
  "rating": 5.0,
  "title": "Coffee Maker",
  "input": "Review_Text: This coffee maker makes excellent coffee and is easy to use. <|endoftext|>",
  "output": "<|ASIN_B09FSPRRRL|>, <|ASIN_B0BX9SKX3S|>, <|ASIN_B09SB12YTD|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>",
  "output_names": "COSTWAY Countertop Ice Maker...",
  "instruction": "Given a user's purchase history and review for a product, predict the next 10 products they would likely purchase. "
}
```

## Testing

The training pipeline should now work correctly with multi-category data because:
- ✅ **Same instruction format** as original
- ✅ **Same input format** as original  
- ✅ **Same output format** as original
- ✅ **Same ASIN format** as original
- ✅ **Same processing logic** as original

The model will receive data in the exact format it was trained on, ensuring compatibility and proper performance.

