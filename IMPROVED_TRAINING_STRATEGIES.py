"""
Improved training strategies for better multimodal model performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImprovedTrainingStrategy:
    """
    Comprehensive training strategy with multiple improvements.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Progressive training
        self.sequence_trainer = SequenceAwareTraining(
            initial_length=config.get('initial_sequence_length', 5),
            final_length=config.get('final_sequence_length', 15),
            num_stages=config.get('progressive_stages', 5)
        )
        
    def setup_training(self, model: nn.Module):
        """Setup training components."""
        self.model = model.to(self.device)
        
        # Optimizer with different learning rates for different components
        param_groups = [
            {'params': model.gpt.parameters(), 'lr': self.config['learning_rate']},
            {'params': model.image_encoder.parameters(), 'lr': self.config['learning_rate'] * 2},
            {'params': model.cross_attention.parameters(), 'lr': self.config['learning_rate'] * 1.5},
            {'params': model.sequence_attention.parameters(), 'lr': self.config['learning_rate'] * 1.5},
            {'params': model.item_prediction_head.parameters(), 'lr': self.config['learning_rate'] * 2},
            {'params': model.stop_prediction_head.parameters(), 'lr': self.config['learning_rate'] * 2}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        if self.config.get('lr_schedule') == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('scheduler_T_0', 100),
                T_mult=2,
                eta_min=self.config.get('min_learning_rate', 1e-6)
            )
        elif self.config.get('lr_schedule') == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['num_epochs'],
                steps_per_epoch=self.config.get('steps_per_epoch', 100)
            )
        
        # Loss function
        self.criterion = ProgressiveLoss(
            item_weight=self.config.get('item_loss_weight', 0.7),
            stop_weight=self.config.get('stop_loss_weight', 0.3),
            contrastive_weight=self.config.get('contrastive_loss_weight', 0.2)
        )
        
    def train_epoch(self, train_loader, val_loader) -> Dict[str, float]:
        """Train for one epoch with improved strategies."""
        self.model.train()
        total_loss = 0
        item_losses = 0
        stop_losses = 0
        contrastive_losses = 0
        
        # Update progressive training stage
        self.sequence_trainer.update_stage(self.current_epoch, self.config['num_epochs'])
        current_length = self.sequence_trainer.get_current_length()
        
        logger.info(f"Epoch {self.current_epoch}: Training with sequence length {current_length}")
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                image_embeddings=batch.get('image_embeddings'),
                attention_mask=batch.get('attention_mask'),
                labels=batch.get('labels'),
                return_contrastive=True
            )
            
            # Compute loss
            loss_dict = self.criterion(outputs, batch)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping', True):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('max_grad_norm', 1.0)
                )
            
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler and self.config.get('lr_schedule') == 'onecycle':
                self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            item_losses += loss_dict['item_loss'].item()
            stop_losses += loss_dict['stop_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'item': f"{loss_dict['item_loss'].item():.4f}",
                'stop': f"{loss_dict['stop_loss'].item():.4f}"
            })
        
        # Validate
        val_metrics = self.validate(val_loader)
        
        # Update learning rate for cosine scheduler
        if self.scheduler and self.config.get('lr_schedule') == 'cosine':
            self.scheduler.step()
        
        # Early stopping
        if val_metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['val_loss']
            self.patience_counter = 0
            # Save best model
            self.save_checkpoint(is_best=True)
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.config.get('early_stopping_patience', 5):
            logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
            return val_metrics
        
        self.current_epoch += 1
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_item_loss': item_losses / len(train_loader),
            'train_stop_loss': stop_losses / len(train_loader),
            **val_metrics
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        item_losses = 0
        stop_losses = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    image_embeddings=batch.get('image_embeddings'),
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels')
                )
                
                loss_dict = self.criterion(outputs, batch)
                total_loss += loss_dict['total_loss'].item()
                item_losses += loss_dict['item_loss'].item()
                stop_losses += loss_dict['stop_loss'].item()
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_item_loss': item_losses / len(val_loader),
            'val_stop_loss': stop_losses / len(val_loader)
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{self.current_epoch}.pth")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, "checkpoints/best_model.pth")
            logger.info(f"Best model saved at epoch {self.current_epoch}")

class CurriculumLearning:
    """
    Curriculum learning strategy that starts with easier examples.
    """
    
    def __init__(self, 
                 difficulty_metric: str = 'sequence_length',
                 start_easy: bool = True):
        self.difficulty_metric = difficulty_metric
        self.start_easy = start_easy
        
    def get_difficulty_score(self, sample: Dict) -> float:
        """Get difficulty score for a sample."""
        if self.difficulty_metric == 'sequence_length':
            return len(sample.get('input_sequence', []))
        elif self.difficulty_metric == 'item_diversity':
            return len(set(sample.get('input_sequence', [])))
        else:
            return 1.0
    
    def sort_samples(self, samples: List[Dict], epoch: int, total_epochs: int) -> List[Dict]:
        """Sort samples by difficulty based on training progress."""
        if self.start_easy:
            # Start with easy samples, gradually include harder ones
            difficulty_threshold = (epoch / total_epochs) * max(
                self.get_difficulty_score(s) for s in samples
            )
            filtered_samples = [
                s for s in samples 
                if self.get_difficulty_score(s) <= difficulty_threshold
            ]
            return sorted(filtered_samples, key=self.get_difficulty_score)
        else:
            # Start with hard samples, gradually include easier ones
            difficulty_threshold = (1 - epoch / total_epochs) * max(
                self.get_difficulty_score(s) for s in samples
            )
            filtered_samples = [
                s for s in samples 
                if self.get_difficulty_score(s) >= difficulty_threshold
            ]
            return sorted(filtered_samples, key=self.get_difficulty_score, reverse=True)

class DataAugmentation:
    """
    Data augmentation strategies for better generalization.
    """
    
    @staticmethod
    def sequence_permutation(sequence: List[str], max_permutations: int = 3) -> List[List[str]]:
        """Create permutations of the input sequence."""
        if len(sequence) <= 1:
            return [sequence]
        
        permutations = [sequence]
        for _ in range(max_permutations):
            perm = sequence.copy()
            # Randomly swap two elements
            i, j = np.random.choice(len(perm), 2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
            permutations.append(perm)
        
        return permutations
    
    @staticmethod
    def item_dropout(sequence: List[str], dropout_rate: float = 0.1) -> List[str]:
        """Randomly drop items from sequence."""
        if len(sequence) <= 1:
            return sequence
        
        num_to_drop = max(1, int(len(sequence) * dropout_rate))
        indices_to_keep = np.random.choice(
            len(sequence), 
            len(sequence) - num_to_drop, 
            replace=False
        )
        
        return [sequence[i] for i in sorted(indices_to_keep)]
    
    @staticmethod
    def sequence_cropping(sequence: List[str], min_length: int = 3) -> List[str]:
        """Randomly crop sequence to different lengths."""
        if len(sequence) <= min_length:
            return sequence
        
        crop_length = np.random.randint(min_length, len(sequence) + 1)
        start_idx = np.random.randint(0, len(sequence) - crop_length + 1)
        
        return sequence[start_idx:start_idx + crop_length]

def main():
    """
    Example usage of improved training strategies.
    """
    logger.info("Setting up improved training strategies...")
    
    config = {
        'learning_rate': 3e-5,
        'num_epochs': 10,
        'weight_decay': 0.01,
        'lr_schedule': 'cosine',
        'gradient_clipping': True,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 5,
        'item_loss_weight': 0.7,
        'stop_loss_weight': 0.3,
        'contrastive_loss_weight': 0.2,
        'initial_sequence_length': 5,
        'final_sequence_length': 15,
        'progressive_stages': 5
    }
    
    # Create training strategy
    training_strategy = ImprovedTrainingStrategy(config)
    
    # Example curriculum learning
    curriculum = CurriculumLearning(difficulty_metric='sequence_length', start_easy=True)
    
    # Example data augmentation
    sample_sequence = ['item1', 'item2', 'item3', 'item4', 'item5']
    augmented_sequences = DataAugmentation.sequence_permutation(sample_sequence)
    logger.info(f"Original: {sample_sequence}")
    logger.info(f"Augmented: {augmented_sequences}")
    
    logger.info("Training strategies setup complete!")

if __name__ == "__main__":
    main()
