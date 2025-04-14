from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

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
        self.special_chars = set(special_chars)  # Convert to set
        self.multimodal_encoder = multimodal_encoder
        self.max_token_length = 1024  # Set this to your model's maximum input length

        # Pre-process text and images into unified token sequences
        self.encoded_sequences = []
        for entry in data:
            # Process text
            instruction_plus_input = processor.format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            
            # Get text tokens
            text_tokens = self.tokenizer.encode(
                f"<|text|> {instruction_plus_input}",
                allowed_special=self.special_chars
            )

            # Process image if available and combine with text tokens
            if entry['local_image_path']:
                image_tensor = self.multimodal_encoder.process_image(entry['local_image_path'])
                if image_tensor is not None:
                    # Get image tokens using vision encoder
                    image_tokens = self.multimodal_encoder.get_image_embedding(image_tensor)
                    # Concatenate text and image tokens directly
                    #combined_tokens = text_tokens + image_tokens.flatten().tolist()
                    combined_tokens = text_tokens + [self.tokenizer.encode('<|image|>')] + image_tokens.flatten().tolist()

                else:
                    combined_tokens = text_tokens
            else:
                combined_tokens = text_tokens

            # Add response tokens at the end
            response_tokens = self.tokenizer.encode(
                f"<|response|> {response_text}",
                #f"{response_text}",
                allowed_special=self.special_chars
            )
            combined_tokens.extend(response_tokens)

            # Log the length of the combined tokens for debugging
            logger.debug(f"Combined token length: {len(combined_tokens)}")

            # Check if the combined token length exceeds the model's context length
            if len(combined_tokens) > self.max_token_length:
                #logger.warning("Combined token length exceeds model's context length. Truncating tokens.")
                combined_tokens = combined_tokens[:self.max_token_length]
            
            self.encoded_sequences.append(combined_tokens)

    def __getitem__(self, index):
        return self.encoded_sequences[index]

    def __len__(self):
        return len(self.data)
