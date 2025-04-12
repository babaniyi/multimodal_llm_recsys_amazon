from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datetime import datetime
from src.data.processor import MultiModalProcessor
#from src.utils.utils import format_input_phi
from tqdm import tqdm

import src.data.processor as processor

class AmazonDatasetAlpaca(Dataset):
    # Adjust Dataset to Alpaca template
    def __init__(self, data, tokenizer, special_chars):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = processor.format_input_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text, allowed_special=special_chars)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class AmazonDatasetPhi(Dataset):
    # Adjust Dataset to Phi3 format with unified multimodal embedding
    def __init__(self, data, tokenizer, special_chars, multimodal_encoder=processor.ImageEncoder()):
        self.data = data
        self.tokenizer = tokenizer
        self.special_chars = special_chars
        self.multimodal_encoder = multimodal_encoder

        # Pre-process text and images into unified token sequences
        self.encoded_sequences = []
        for entry in data:
            # Process text
            instruction_plus_input = processor.format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            
            # Get text tokens
            text_tokens = self.tokenizer.encode(
                instruction_plus_input,
                allowed_special=special_chars
            )

            # Process image if available and combine with text tokens
            if entry['local_image_path']:
                image_tensor = self.multimodal_encoder.process_image(entry['local_image_path'])
                if image_tensor is not None:
                    # Get image tokens using vision encoder
                    image_tokens = self.multimodal_encoder.get_image_embedding(image_tensor)
                    # Concatenate text and image tokens directly
                    combined_tokens = text_tokens + image_tokens.flatten().tolist()
                else:
                    combined_tokens = text_tokens
            else:
                combined_tokens = text_tokens

            # Add response tokens at the end
            response_tokens = self.tokenizer.encode(
                response_text,
                allowed_special=special_chars
            )
            combined_tokens.extend(response_tokens)
            
            self.encoded_sequences.append(combined_tokens)

    def __getitem__(self, index):
        return self.encoded_sequences[index]

    def __len__(self):
        return len(self.data)


class MultiModalAmazonDataset(Dataset):
    def __init__(
        self, 
        data: List[Dict], 
        tokenizer, 
        special_chars: List[str],
        multimodal_processor: MultiModalProcessor,
        max_sequence_length: int = 50,
        min_sequence_length: int = 2,
        prediction_window: int = 10  # Number of future items to predict
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.special_chars = special_chars
        self.multimodal_processor = multimodal_processor
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.prediction_window = prediction_window
        
        # Pre-process all sequences
        self.processed_sequences = []
        
        # Group data by user and sort by timestamp
        user_sequences = {}
        for entry in data:
            user_id = entry['user_id']
            if user_id not in user_sequences:
                user_sequences[user_id] = []
            user_sequences[user_id].append(entry)
            
        # Process each user's sequence
        for user_id, sequence in tqdm(user_sequences.items(), desc="Processing user sequences"):
            # Sort by timestamp
            sequence = sorted(sequence, key=lambda x: x['timestamp'])
            
            # Create sliding windows
            for i in range(0, len(sequence) - self.min_sequence_length):
                end_idx = min(i + self.max_sequence_length, len(sequence))
                target_end_idx = min(end_idx + self.prediction_window, len(sequence))
                
                if end_idx - i < self.min_sequence_length:
                    continue
                    
                # Get input sequence and target sequence
                input_sequence = sequence[i:end_idx]
                target_sequence = sequence[end_idx:target_end_idx]
                
                # Process input sequence with temporal features
                processed_sequence, temporal_features = self.multimodal_processor.process_sequence(input_sequence)
                
                # Format text with product and temporal information
                instruction = self._format_instruction(processed_sequence, temporal_features)
                target_text = self._format_target(target_sequence)
                
                # Format full text
                instruction_plus_input = processor.format_input_phi({
                    'instruction': instruction,
                    'input': ''
                })
                response_text = f"\n<|assistant|>:\n{target_text}"
                full_text = instruction_plus_input + response_text
                
                # Tokenize text
                encoded_text = self.tokenizer.encode(full_text, allowed_special=self.special_chars)
                
                self.processed_sequences.append({
                    'text_tokens': encoded_text,
                    'user_id': user_id,
                    'temporal_features': temporal_features
                })

    def _format_instruction(self, processed_sequence: List[Dict], temporal_features: Dict) -> str:
        """Format instruction with product history and temporal insights."""
        # Format product history
        products_desc = []
        for item in processed_sequence:
            desc = (
                f"Product: {item['title']}\n"
                f"Category: {item['category']}\n"
                f"Price: {item['price']}\n"
                f"Rating: {item['rating']} ({item['review_count']} reviews)\n"
                f"Purchased on: {datetime.fromtimestamp(item['timestamp']).strftime('%Y-%m-%d %H:%M')}"
            )
            if item.get('features'):
                desc += f"\nFeatures: {', '.join(item['features'])}"
            products_desc.append(desc)
            
        history = "\n\n".join(products_desc)
        
        # Format temporal insights
        temporal_insights = (
            f"Purchase patterns:\n"
            f"- Average time between purchases: {temporal_features['avg_time_between_purchases'] / 86400:.1f} days\n"
            f"- Most common purchase days: {', '.join(map(lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x], temporal_features['common_purchase_days']))}\n"
            f"- Most common purchase hours: {', '.join(map(str, temporal_features['common_purchase_hours']))}\n"
            f"- Total purchases: {temporal_features['total_purchases']}"
        )
        
        return (
            f"Based on the user's purchase history:\n\n"
            f"{'-' * 40}\n"
            f"{history}\n"
            f"{'-' * 40}\n\n"
            f"{temporal_insights}\n\n"
            f"Predict the next {self.prediction_window} items they are likely to purchase."
        )

    def _format_target(self, target_sequence: List[Dict]) -> str:
        """Format the target sequence as a response."""
        products = []
        for item in target_sequence:
            products.append(f"<|item_{item['asin']}|>")
            
        # Pad with endoftext tokens if needed
        while len(products) < self.prediction_window:
            products.append("<|endoftext|>")
            
        return ", ".join(products)

    def __getitem__(self, index):
        return self.processed_sequences[index]

    def __len__(self):
        return len(self.processed_sequences)

