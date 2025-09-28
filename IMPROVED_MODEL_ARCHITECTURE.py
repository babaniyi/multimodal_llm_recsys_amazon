"""
Improved model architecture for better multimodal recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ImprovedMultimodalGPT(nn.Module):
    """
    Enhanced multimodal GPT model with better fusion and sequence awareness.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Base GPT model with larger context
        gpt_config = GPT2Config(
            vocab_size=config['vocab_size'],
            n_positions=config['max_seq_len'],
            n_ctx=config['max_seq_len'],
            n_embd=config['d_model'],
            n_layer=config['num_layers'],
            n_head=config['num_heads'],
            dropout=config['dropout'],
            activation_function="gelu_new"
        )
        
        self.gpt = GPT2Model(gpt_config)
        
        # Image encoder with better architecture
        self.image_encoder = nn.Sequential(
            nn.Linear(config['image_embedding_dim'], 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, config['d_model']),
            nn.LayerNorm(config['d_model'])
        )
        
        # Cross-attention for multimodal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'],
            num_heads=config['num_heads'],
            dropout=0.1,
            batch_first=True
        )
        
        # Sequence-aware attention
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'],
            num_heads=config['num_heads'],
            dropout=0.1,
            batch_first=True
        )
        
        # Progressive prediction heads
        self.item_prediction_head = nn.Linear(config['d_model'], config['vocab_size'])
        self.stop_prediction_head = nn.Linear(config['d_model'], 2)  # Binary: continue/stop
        
        # Contrastive learning components
        self.contrastive_projection = nn.Linear(config['d_model'], 256)
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, 
                input_ids: torch.Tensor,
                image_embeddings: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_contrastive: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with improved multimodal fusion.
        """
        batch_size, seq_len = input_ids.shape
        
        # Get text embeddings
        text_embeddings = self.gpt.embeddings.word_embeddings(input_ids)
        text_embeddings = text_embeddings + self.gpt.embeddings.position_embeddings(
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        )
        
        # Process image embeddings if available
        if image_embeddings is not None:
            image_embeddings = self.image_encoder(image_embeddings)
            
            # Cross-attention between text and images
            attended_text, _ = self.cross_attention(
                text_embeddings, image_embeddings, image_embeddings
            )
            text_embeddings = text_embeddings + attended_text
        
        # Sequence-aware attention
        sequence_output, _ = self.sequence_attention(
            text_embeddings, text_embeddings, text_embeddings,
            key_padding_mask=attention_mask
        )
        
        # Pass through GPT layers
        hidden_states = sequence_output
        for layer in self.gpt.h:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        # Progressive prediction
        item_logits = self.item_prediction_head(hidden_states)
        stop_logits = self.stop_prediction_head(hidden_states)
        
        outputs = {
            'item_logits': item_logits,
            'stop_logits': stop_logits,
            'hidden_states': hidden_states
        }
        
        # Contrastive learning
        if return_contrastive:
            contrastive_features = self.contrastive_projection(hidden_states)
            outputs['contrastive_features'] = contrastive_features
        
        return outputs

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for better item representation learning.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, 
                user_features: torch.Tensor,
                item_features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between user and item features.
        """
        # Normalize features
        user_features = F.normalize(user_features, dim=-1)
        item_features = F.normalize(item_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(user_features, item_features.T) / self.temperature
        
        # Compute loss
        loss = F.cross_entropy(similarity, labels)
        return loss

class ProgressiveLoss(nn.Module):
    """
    Progressive loss that balances item prediction and stop prediction.
    """
    
    def __init__(self, 
                 item_weight: float = 0.7,
                 stop_weight: float = 0.3,
                 contrastive_weight: float = 0.2):
        super().__init__()
        self.item_weight = item_weight
        self.stop_weight = stop_weight
        self.contrastive_weight = contrastive_weight
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute progressive loss.
        """
        # Item prediction loss
        item_loss = F.cross_entropy(
            outputs['item_logits'].view(-1, outputs['item_logits'].size(-1)),
            labels['item_labels'].view(-1),
            ignore_index=-100
        )
        
        # Stop prediction loss
        stop_loss = F.cross_entropy(
            outputs['stop_logits'].view(-1, 2),
            labels['stop_labels'].view(-1)
        )
        
        # Total loss
        total_loss = (self.item_weight * item_loss + 
                     self.stop_weight * stop_loss)
        
        # Add contrastive loss if available
        if 'contrastive_loss' in outputs:
            total_loss += self.contrastive_weight * outputs['contrastive_loss']
        
        return {
            'total_loss': total_loss,
            'item_loss': item_loss,
            'stop_loss': stop_loss
        }

class SequenceAwareTraining:
    """
    Sequence-aware training strategy that progressively increases sequence length.
    """
    
    def __init__(self, 
                 initial_length: int = 5,
                 final_length: int = 15,
                 num_stages: int = 5):
        self.initial_length = initial_length
        self.final_length = final_length
        self.num_stages = num_stages
        self.current_stage = 0
        
    def get_current_length(self) -> int:
        """Get current sequence length based on training stage."""
        progress = self.current_stage / self.num_stages
        return int(self.initial_length + 
                  progress * (self.final_length - self.initial_length))
    
    def update_stage(self, epoch: int, total_epochs: int):
        """Update training stage based on epoch."""
        self.current_stage = int((epoch / total_epochs) * self.num_stages)

def create_improved_model(config: Dict) -> ImprovedMultimodalGPT:
    """
    Create an improved multimodal model with better architecture.
    """
    model = ImprovedMultimodalGPT(config)
    
    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    return model

def main():
    """
    Example usage of improved model architecture.
    """
    logger.info("Creating improved multimodal model...")
    
    config = {
        'vocab_size': 50257,
        'max_seq_len': 1024,
        'd_model': 768,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'image_embedding_dim': 2048
    }
    
    model = create_improved_model(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    image_embeddings = torch.randn(batch_size, seq_len, config['image_embedding_dim'])
    
    outputs = model(input_ids, image_embeddings, return_contrastive=True)
    logger.info(f"Output shapes: {[(k, v.shape) for k, v in outputs.items()]}")

if __name__ == "__main__":
    main()
