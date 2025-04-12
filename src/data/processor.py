import pandas as pd
import torch
import math
import os
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Union
import requests
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# DATA PROCESSING UTILITIES:
# ==========================================

def cast_unix_to_date(unix_time_str: pd.Series) -> pd.Series:
    """
    Converts a Series of Unix timestamps (milliseconds) to datetime strings.

    Args:
        unix_time_str (pd.Series): Series of Unix timestamps as strings.

    Returns:
        pd.Series: Series of datetime strings in the format 'YYYY-MM-DD'.
    """

    # Convert to numeric directly
    unix_timestamps = pd.to_numeric(unix_time_str)

    # Convert to datetime using NumPy for efficiency
    datetime_values = pd.to_datetime(unix_timestamps / 1000, unit='s')  # Convert milliseconds to seconds

    # Format as strings
    return datetime_values.dt.strftime('%Y-%m-%d')

def filter_users_by_interactions(review_data: List[Dict], min_interactions: int = 3) -> List[Dict]:
    """Filter users to keep only those with at least min_interactions."""
    # Count interactions per user
    user_interactions = defaultdict(int)
    for review in review_data:
        user_id = review.get('user_id')
        if user_id:
            user_interactions[user_id] += 1
    
    # Filter users with enough interactions
    valid_users = {user_id for user_id, count in user_interactions.items() 
                  if count >= min_interactions}
    
    # Filter review data
    filtered_data = [review for review in review_data 
                    if review.get('user_id') in valid_users]
    
    logger.info(f"Original users: {len(user_interactions)}")
    logger.info(f"Users with >= {min_interactions} interactions: {len(valid_users)}")
    logger.info(f"Original reviews: {len(review_data)}")
    logger.info(f"Filtered reviews: {len(filtered_data)}")
    
    return filtered_data


def get_next_items(
    df: pd.DataFrame,
    padding_strategy='repeat',
    pad_token="<|endoftext|>",
    number_of_items_to_predict=5,
    min_interactions=3
):
    """
    For each row in the dataframe, get the next N items the user interacted with.
    
    Args:
        df (pd.DataFrame): Must have 'user_id', 'timestamp', 'parent_asin', 'title'
        padding_strategy (str): 'none', 'repeat', 'pad_token'
        pad_token (str): Value to pad with if applicable
        number_of_items_to_predict (int): Number of items to get
        min_interactions (int): Minimum interactions per user
    
    Returns:
        pd.DataFrame with ['user_id', 'parent_asin', 'timestamp', 'next_items', 'next_item_names']
    """
    df_sorted = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    next_items_data = []

    for user_id, user_data in df_sorted.groupby('user_id'):
        user_data = user_data.reset_index(drop=True)
        num_interactions = len(user_data)

        if num_interactions < min_interactions:
            continue

        for i in range(num_interactions):
            # Get next items, but don't skip if there aren't enough
            next_df = user_data.loc[i + 1:i + number_of_items_to_predict, :]
            next_items = next_df['parent_asin'].tolist()
            next_item_names = next_df['title'].tolist()

            # Handle padding if needed
            if len(next_items) < number_of_items_to_predict:
                num_missing = number_of_items_to_predict - len(next_items)

                if padding_strategy == 'repeat':
                    if len(next_items) > 0:
                        next_items += [next_items[-1]] * num_missing
                        next_item_names += [next_item_names[-1]] * num_missing
                    else:
                        # Can't repeat from an empty list, fallback to pad_token
                        next_items += [pad_token] * number_of_items_to_predict
                        next_item_names += [pad_token] * number_of_items_to_predict

                elif padding_strategy == 'pad_token':
                    next_items += [pad_token] * num_missing
                    next_item_names += [pad_token] * num_missing

                elif padding_strategy == 'none':
                    # Skip rows where next_10_items do not exist
                    continue

            next_items_data.append((
                user_data.loc[i, 'user_id'],
                user_data.loc[i, 'parent_asin'],
                user_data.loc[i, 'date'],
                ", ".join(next_items),
                "| ".join(next_item_names)
            ))

    df_next_items = pd.DataFrame(
        next_items_data,
        columns=['user_id', 'parent_asin', 'date', 'next_items', 'next_item_names']
    )

    return df_next_items


def details_to_sentence(details_dict):
    sentences = [f"{key.title()}: {value}\n" for key, value in details_dict.items()]
    history = "\n\n".join(sentences)
    return history
    

def format_input_alpaca(entry):
    """ 
    Format entry according to the Alpaca-style prompt template. Example of data entry is as follows:

    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Identify the correct spelling of the following word.

    ### Input:
    Occassion

    ### Response:
    The correct spelling is 'Occasion.'
    """
    
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def format_input_phi(entry):
    """ 
    Phi-3 prompt template
    This prompt template is substantially shorter, which reduces the runtime and hardware requirements for 
    finetuning the LLM and generating text since the input prompts are shorter. 
    It formats the data entry as follows:

    <user>
    Identify the correct spelling of the following word: 'Occasion'

    <assistant>
    The correct spelling is 'Occasion'.
    """ 

    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def split_data_temporal(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the data into train, validation, and test sets while preserving the temporal order of each user's interactions.

    Args:
    - df (pd.DataFrame): DataFrame containing user interactions with 'user_id' and 'date' columns.
    - train_ratio (float): Proportion of data to be used for training.
    - val_ratio (float): Proportion of data to be used for validation.
    - test_ratio (float): Proportion of data to be used for testing.

    Returns:
    - train_df (pd.DataFrame): Training set.
    - val_df (pd.DataFrame): Validation set.
    - test_df (pd.DataFrame): Test set.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1."

    # Sort data by user_id and date to maintain chronological order
    df_sorted = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)

    # Create empty DataFrames for train, validation, and test
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Split each user's data chronologically
    for user_id, group in df_sorted.groupby('user_id'):
        num_interactions = len(group)

        # Calculate indices for train, validation, and test splits
        train_end = int(train_ratio * num_interactions)
        val_end = train_end + int(val_ratio * num_interactions)

        # Ensure there's at least one interaction in each split if possible
        if train_end == 0:
            train_end = 1
        if val_end == train_end:
            val_end = train_end + 1

        # Concatenate each user's splits to the respective DataFrame
        train_df = pd.concat([train_df, group.iloc[:train_end]])
        val_df = pd.concat([val_df, group.iloc[train_end:val_end]])
        test_df = pd.concat([test_df, group.iloc[val_end:]])

    # Reset index after concatenation
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_split_data(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], output_dir: str):
    """
    Save split data to JSON files.
    
    Args:
        train_data (List[Dict]): Training data
        val_data (List[Dict]): Validation data
        test_data (List[Dict]): Test data
        output_dir (str): Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f)
        
    logger.info(f"Saved split data to {output_dir}")

def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    file_path_str = str(file_path)  # Convert Path to string
    with open(file_path_str, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(file_path_str)}"):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {e}")
                continue
    return data

def process_review_data(review_data: List[Dict]) -> pd.DataFrame:
    """Process review data into a DataFrame."""
    processed_data = []
    for review in tqdm(review_data, desc="Processing review data"):
        processed_review = {
            'user_id': review.get('user_id'),
            'parent_asin': review.get('parent_asin'),
            'rating': review.get('rating'),
            'review_text': review.get('text', ''),
            'review_title': review.get('title', ''),
            'timestamp': review.get('timestamp'),
            'verified_purchase': review.get('verified_purchase', False),
            'helpful_votes': review.get('helpful_vote', 0)
        } # [user_id, parent_asin, rating, review_text, review_title, timestamp, verified_purchase, helpful_votes]
        processed_data.append(processed_review)
    return pd.DataFrame(processed_data)

def is_valid_image_url(url: str) -> bool:
    """Check if URL is valid and points to an image."""
    try:
        response = requests.head(url, timeout=5)
        content_type = response.headers.get('content-type', '')
        return response.status_code == 200 and 'image' in content_type.lower()
    except:
        return False

def download_image(url: str, cache_dir: Path, filename: str) -> Optional[str]:
    """Download and cache an image."""
    if not url:
        return None
        
    cache_path = cache_dir / filename
    
    # Check if image already exists in cache
    if cache_path.exists():
        try:
            # Verify the image is valid
            with Image.open(cache_path) as img:
                img.verify()
            return str(cache_path)
        except Exception as e:
            logger.warning(f"Found corrupted cached image {filename}, will redownload: {str(e)}")
            # Remove corrupted image
            cache_path.unlink()
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save image
        with open(cache_path, 'wb') as f:
            f.write(response.content)
            
        # Verify the downloaded image
        with Image.open(cache_path) as img:
            img.verify()
            
        return str(cache_path)
    except Exception as e:
        logger.error(f"Error downloading image {url}: {str(e)}")
        if cache_path.exists():
            cache_path.unlink()
        return None

def process_metadata(metadata: List[Dict], cache_dir: Path) -> pd.DataFrame:
    """Process metadata into a DataFrame with image validation and caching."""
    processed_metadata = []
    invalid_images = []
    
    for item in tqdm(metadata, desc="Processing metadata"):
        # Get the main product image URL (highest quality)
        images = item.get('images', [])
        main_image_url = None
        local_image_path = None
        parent_asin = item.get('parent_asin')
        
        # Skip items without parent_asin
        if not parent_asin:
            logger.warning("Skipping item without parent_asin")
            continue
        
        for img in images:
            if img.get('variant') == 'MAIN':
                main_image_url = img.get('hi_res')
                break
        
        # Validate and cache image if URL exists
        if main_image_url:
            if is_valid_image_url(main_image_url):
                local_image_path = download_image(
                    main_image_url,
                    cache_dir,
                    f"{parent_asin}.jpg"
                )
            else:
                invalid_images.append({
                    'parent_asin': parent_asin,
                    'url': main_image_url
                })
        
        processed_item = {
            'parent_asin': parent_asin,
            'title': item.get('title', ''),
            #'description': ' '.join(item.get('description', [])),
            #'features': ' '.join(item.get('features', [])),
            'price': item.get('price'),
            'main_category': item.get('main_category'),
            #'average_rating': item.get('average_rating'),
            #'rating_number': item.get('rating_number'),
            'store': item.get('store'),
            'brand': item.get('details', {}).get('Brand', ''),
            'image_url': main_image_url,
            'local_image_path': local_image_path
        }# ['parent_asin', 'title', 'price', 'main_category', 'store', 'brand', 'image_url', 'local_image_path']
        processed_metadata.append(processed_item)
    
    # Log invalid images 
    if invalid_images:
        logger.warning(f"Found {len(invalid_images)} invalid images")
        invalid_images_path = cache_dir.parent / "invalid_images.jsonl"
        with open(invalid_images_path, 'w') as f:
            for item in invalid_images:
                f.write(json.dumps(item) + '\n')
    
    return pd.DataFrame(processed_metadata)

def merge_data(review_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Merge review data with metadata using parent_asin."""
    logger.info("Merging review data with metadata...")
    merged_df = pd.merge(
        review_df,
        metadata_df,
        on='parent_asin',
        how='inner'
    )
    logger.info(f"Total records after merge: {len(merged_df)}")
    return merged_df


def save_processed_data(data: pd.DataFrame, output_path: Union[str, Path]):
    """Save processed data to JSONL file."""
    output_path_str = str(output_path)  # Convert Path to string
    logger.info(f"Saving processed data to {output_path_str}")
    data.to_json(output_path_str, orient='records', lines=True)
    logger.info("Data saved successfully")





class ImageEncoder:
    def __init__(self, image_size: int = 224):
        """
        Initialize the multi-modal processor for handling both text and image data.
        
        Args:
            image_size (int): Size to resize images to
        """
        self.image_size = image_size
        
        # Initialize image processor and model
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.vision_model = AutoModel.from_pretrained("microsoft/resnet-50")

        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
            
    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Process an image into a tensor suitable for the model.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[torch.Tensor]: Processed image tensor if successful, None otherwise
        """
        try:
            # Load and transform image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Apply transformations
                image_tensor = self.transform(img)
                # Ensure we have a tensor
                if not isinstance(image_tensor, torch.Tensor):
                    logger.warning(f"Transform did not return a tensor for {image_path}")
                    return None
                return image_tensor
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
            
    def get_image_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding for an image using the vision model.
        
        Args:
            image_tensor (torch.Tensor): Processed image tensor
            
        Returns:
            torch.Tensor: Image embedding
        """
        # with torch.no_grad():
        #     outputs = self.vision_model(image_tensor.unsqueeze(0))
        #     return outputs.last_hidden_state.mean(dim=1)

        with torch.no_grad():
            # Get the output from the vision model
            outputs = self.vision_model(image_tensor.unsqueeze(0))
            
            # ResNet-50 outputs a 7x7 feature map in the last hidden state
            # We need to get the final pooling layer output
            if hasattr(self.vision_model, 'avgpool'):
                # If we have access to the pooling layer, use it
                pooled_output = self.vision_model.avgpool(outputs.last_hidden_state)
                # Flatten the output
                flattened_output = torch.flatten(pooled_output, 1)
                return flattened_output
            else:
                # If we don't have direct access to the pooling layer,
                # use global average pooling on the last hidden state
                # Shape: [batch_size, 7, 7, channels]
                pooled_output = torch.mean(outputs.last_hidden_state, dim=[2, 3])
                return pooled_output



## MODEL UTILITIES:
# ==========================================

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)
