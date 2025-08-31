from tqdm import tqdm
import logging
from pathlib import Path
import json
import numpy as np
import tiktoken
import os

from ..utils.utils import setup_logging, set_seed, load_config
from . import processor

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)
set_seed(42)

# Test the logger
logger.info("Starting data preparation script")

## TODO: 
# - Add a function to get the next 10 items for each user
# - USE Multi-head attention with PyTorch's scaled dot product attention and FlashAttention



def main():

    # STEP 0: Load data
    # ==========================================
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    cache_dir = processed_dir / "image_cache"


    # Create necessary directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    # Test mode parameters
    TEST_MODE = config['model_config']['test_mode']
    #TEST_MODE = TEST_MODE.lower()
    #TEST_MODE = json.loads(TEST_MODE)
    TEST_SIZE = int(config['model_config']['test_size'])   # Number of items to process in test mode

    # Load data
    logger.info("Loading review data...")
    review_data = processor.load_jsonl(raw_dir / "Appliances.jsonl")


    if TEST_MODE:
        logger.info(f"Running in test mode with {TEST_SIZE} items")
        # Take a random sample of reviews
        review_data = review_data[:TEST_SIZE]

    # Filter users with at least 2 interactions
    logger.info(f"Filtering users with at least {config['data_config']['min_interactions']} interactions...")
    filtered_review_data = processor.filter_users_by_interactions(review_data, min_interactions=config['data_config']['min_interactions'])

    # Get unique parent_asins from filtered reviews
    valid_parent_asins = set(review.get('parent_asin') for review in filtered_review_data)
    
    logger.info("Loading metadata...")
    metadata = processor.load_jsonl(raw_dir / "meta_Appliances.jsonl")


    if TEST_MODE:
        # Filter metadata to only include items from test reviews
        metadata = [item for item in metadata 
                   if item.get('parent_asin') in valid_parent_asins]
        metadata = metadata[:TEST_SIZE]


    # STEP 1: Filtering and processing data
    # ==========================================

    # > Filter metadata to only include items from filtered reviews
    filtered_metadata = [item for item in metadata 
                        if item.get('parent_asin') in valid_parent_asins]
    logger.info(f"Filtered metadata items: {len(filtered_metadata)}")

    # 1.01 Process reviews and metadata
    # Process data
    logger.info("Processing review data...")
    review_df = processor.process_review_data(filtered_review_data)
    
    logger.info("Processing metadata and caching images...")
    metadata_df = processor.process_metadata(filtered_metadata, cache_dir)
    
    # Final merge data: review and metadata
    merged_df = processor.merge_data(review_df, metadata_df)

    logger.info(f"Final merged dataframe shape: {merged_df.shape}")

    
    # 1.02 Data user and item to custom format
    # Prepare the data by setting the user_id and item_id to a special token format
    # ==========================================

    #cols_to_drop = ["timestamp", "parent_asin", "user_id", "details", "local_image_path"]
    
    review_cols = ["review_text"]
    item_cols = []
    
    merged_df["date"] = processor.cast_unix_to_date(merged_df["timestamp"])

    merged_df = merged_df.sort_values(by=["date", "user_id"])
    #merged_df['parent_asin_original'] = merged_df['parent_asin'].copy()
    merged_df["parent_asin"] = "<|ASIN_" + merged_df["parent_asin"] + "|>"
    #merged_df['input'] = merged_df[review_cols + item_cols].apply(details_to_sentence)
    merged_df["input"] = merged_df[review_cols + item_cols].apply(lambda x: '\n'.join(f"{col.title()}: {value} <|endoftext|>" for col, value in x.items()), axis=1)

   

    # > Prepare the data labels: Extract the next items purchased by the user
    labels_df = processor.get_next_items(
        df=merged_df, padding_strategy='pad_token', pad_token="<|endoftext|>", 
        number_of_items_to_predict=config['data_config']['number_of_items_to_predict'], 
        min_interactions=config['data_config']['min_interactions'])
    
    join_keys = ["user_id", "parent_asin", "date"]
    merged_df = merged_df.merge(labels_df, on=join_keys, how="inner")

    #_________
    # Save parent_asin and product title to JSON
    parent_asin_title_path = processed_dir / "parent_asin_title.json"
    parent_asin_title_data = merged_df[['parent_asin', 'title']].to_dict(orient='records')
    with open(parent_asin_title_path, 'w') as f:
        json.dump(parent_asin_title_data, f)
    logger.info(f"Saved parent_asin and product title to {parent_asin_title_path}")

    #__________

    merged_df = merged_df.drop(columns=review_cols + item_cols, axis=1)
    merged_df = merged_df.drop(columns=["timestamp"], axis=1)

    del labels_df

    logger.info(f"Finished processing data and collecting labels: {merged_df.shape}")

    # 1.03 Tokenizer
    #        Lets the identify user_id, item and "<|endoftext|>" as additional tokens
    #_______________________________________________________________________________
    special_user_item_ids = ["<|endoftext|>"] + merged_df["parent_asin"].unique().tolist()
    special_user_item_ids = set(special_user_item_ids)

    #tokenizer = tiktoken.get_encoding("gpt2")   

    # Add special tokens to the tokenizer
    #end_of_text_id = np.array(tokenizer.encode("<|endoftext|>", allowed_special=special_user_item_ids)).item()


    # 2. DATA PREP - Data Loaders, Custom collate, etc.
    # ==========================================

    ## 2.01 Data preparation - input, output and instruction
    #_______________________________________________________________________________

    df_model_ready = merged_df.copy()

    df_model_ready = df_model_ready.rename(columns={'next_items': "output", 'next_item_names': "output_names"})
    df_model_ready['instruction'] = config['data_config']['user_prompt']


    # Save processed data with test mode indicator
    output_filename = "processed_data_test.jsonl" if TEST_MODE else "processed_data.jsonl"
    output_path = processed_dir / output_filename
    processor.save_processed_data(df_model_ready, output_path)
    logger.info(f"Saved processed data to {output_path}")

    # 2.02 Split to train, validation and test
    #_______________________________________________________________________________
    train_df, val_df, test_df = processor.split_data_temporal(
        df_model_ready, 
        train_ratio=config['data_split_ratio']['train_ratio'], 
        val_ratio=config['data_split_ratio']['val_ratio'], 
        test_ratio=config['data_split_ratio']['test_ratio'])

    train_data = train_df.to_dict(orient="records")
    test_data = test_df.to_dict(orient="records")
    val_data = val_df.to_dict(orient="records")

    del train_df, val_df, test_df

    processor.save_split_data(train_data, val_data, test_data, str(processed_dir))

    special_user_item_ids_path = processed_dir / "special_user_item_ids.json"    
    
    with open(special_user_item_ids_path, "w") as f:
        json.dump(list(special_user_item_ids), f)
    logger.info(f"Saved special_user_item_ids to {special_user_item_ids_path}")



    # Log split statistics
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")
    logger.info(f"Test set size: {len(test_data)}")
    
    logger.info("Data preparation and splitting completed successfully!")







if __name__ == "__main__":
    main()
