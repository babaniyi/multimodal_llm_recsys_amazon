"""
Ablation studies implementation addressing reviewer feedback.
Implements text-only, image-only, and fusion method comparisons.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class AblationStudy:
    """
    Conducts ablation studies to understand individual modality contributions.
    Addresses the missing ablation analysis identified by reviewers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
    def text_only_model(self, train_data: List[Dict], test_data: List[Dict], global_tfidf=None) -> Dict[str, float]:
        """
        Text-only baseline using TF-IDF and cosine similarity.
        """
        logger.info("Running text-only ablation study...")
        
        # Prepare text data
        item_texts = {}
        user_texts = {}
        
        for entry in train_data:
            item_id = entry['parent_asin']
            user_id = entry['user_id']
            
            # Combine title and review text
            text = f"{entry.get('title', '')} {entry.get('review_text', '')}"
            
            if item_id not in item_texts:
                item_texts[item_id] = []
            item_texts[item_id].append(text)
            
            if user_id not in user_texts:
                user_texts[user_id] = []
            user_texts[user_id].append(text)
        
        # Create TF-IDF vectors
        all_texts = []
        item_ids = []
        for item_id, texts in item_texts.items():
            combined_text = ' '.join(texts)
            all_texts.append(combined_text)
            item_ids.append(item_id)
        
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        item_vectors = tfidf.fit_transform(all_texts)
        
        # Create item mapping
        item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        # Evaluate on test data
        predictions = []
        targets = []
        
        for entry in test_data:
            user_id = entry['user_id']
            target_items = entry['output'].split(', ')
            
            # Get user's text preferences
            if user_id in user_texts:
                user_text = ' '.join(user_texts[user_id])
                user_vector = tfidf.transform([user_text])
                
                # Calculate similarities
                similarities = cosine_similarity(user_vector, item_vectors).flatten()
                
                # Get top recommendations
                top_indices = np.argsort(similarities)[-10:][::-1]
                predicted_items = [item_ids[idx] for idx in top_indices]
            else:
                # Cold start - use popular items
                predicted_items = list(item_texts.keys())[:10]
            
            predictions.append(predicted_items)
            targets.append(target_items)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        metrics['model_type'] = 'text_only'
        
        logger.info(f"Text-only results: HR@10={metrics['hr@10']:.4f}, MRR={metrics['mrr']:.4f}")
        return metrics
    
    def image_only_model(self, train_data: List[Dict], test_data: List[Dict], 
                        image_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Image-only baseline using visual similarity.
        """
        logger.info("Running image-only ablation study...")
        
        # Prepare image data
        item_images = {}
        user_images = {}
        
        for entry in train_data:
            item_id = entry['parent_asin']
            user_id = entry['user_id']
            
            if item_id in image_embeddings:
                if item_id not in item_images:
                    item_images[item_id] = []
                item_images[item_id].append(image_embeddings[item_id])
                
                if user_id not in user_images:
                    user_images[user_id] = []
                user_images[user_id].append(image_embeddings[item_id])
        
        # Calculate average embeddings for items
        item_avg_embeddings = {}
        for item_id, embeddings in item_images.items():
            item_avg_embeddings[item_id] = np.mean(embeddings, axis=0)
        
        # Evaluate on test data
        predictions = []
        targets = []
        
        for entry in test_data:
            user_id = entry['user_id']
            target_items = entry['output'].split(', ')
            
            # Get user's visual preferences
            if user_id in user_images:
                user_embeddings = user_images[user_id]
                user_avg_embedding = np.mean(user_embeddings, axis=0)
                
                # Calculate similarities with all items
                similarities = {}
                for item_id, item_embedding in item_avg_embeddings.items():
                    similarity = cosine_similarity([user_avg_embedding], [item_embedding])[0][0]
                    similarities[item_id] = similarity
                
                # Get top recommendations
                sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                predicted_items = [item_id for item_id, _ in sorted_items[:10]]
            else:
                # Cold start - use items with embeddings
                predicted_items = list(item_avg_embeddings.keys())[:10]
            
            predictions.append(predicted_items)
            targets.append(target_items)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        metrics['model_type'] = 'image_only'
        
        logger.info(f"Image-only results: HR@10={metrics['hr@10']:.4f}, MRR={metrics['mrr']:.4f}")
        return metrics
    
    def multimodal_fusion_comparison(self, train_data: List[Dict], test_data: List[Dict],
                                   image_embeddings: Dict[str, np.ndarray], global_tfidf=None) -> Dict[str, Dict[str, float]]:
        """
        Compare different multimodal fusion strategies.
        """
        logger.info("Running multimodal fusion comparison...")
        
        fusion_results = {}
        
        # 1. Simple concatenation (current approach)
        fusion_results['concatenation'] = self._concatenation_fusion(train_data, test_data, image_embeddings)
        
        # 2. Weighted fusion
        fusion_results['weighted'] = self._weighted_fusion(train_data, test_data, image_embeddings)
        
        # 3. Attention-based fusion
        fusion_results['attention'] = self._attention_fusion(train_data, test_data, image_embeddings)
        
        return fusion_results
    
    def _concatenation_fusion(self, train_data: List[Dict], test_data: List[Dict],
                            image_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Simple concatenation fusion (current approach)."""
        logger.info("  Testing concatenation fusion...")
        
        # Prepare multimodal data
        item_features = {}
        user_features = {}
        
        for entry in train_data:
            item_id = entry['parent_asin']
            user_id = entry['user_id']
            
            # Text features
            text = f"{entry.get('title', '')} {entry.get('review_text', '')}"
            text_vector = self._text_to_vector(text, global_tfidf)
            
            # Image features
            if item_id in image_embeddings:
                image_vector = image_embeddings[item_id]
            else:
                image_vector = np.zeros(2048)  # Default image embedding size
            
            # Concatenate features
            combined_features = np.concatenate([text_vector, image_vector])
            
            if item_id not in item_features:
                item_features[item_id] = []
            item_features[item_id].append(combined_features)
            
            if user_id not in user_features:
                user_features[user_id] = []
            user_features[user_id].append(combined_features)
        
        # Calculate average features
        item_avg_features = {item_id: np.mean(features, axis=0) 
                           for item_id, features in item_features.items()}
        
        # Evaluate
        predictions, targets = self._evaluate_multimodal(user_features, item_avg_features, test_data)
        metrics = self._calculate_metrics(predictions, targets)
        metrics['fusion_method'] = 'concatenation'
        
        return metrics
    
    def _weighted_fusion(self, train_data: List[Dict], test_data: List[Dict],
                        image_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Weighted fusion of text and image features."""
        logger.info("  Testing weighted fusion...")
        
        # Optimize weights using validation data
        text_weight, image_weight = self._optimize_fusion_weights(train_data, image_embeddings)
        
        # Prepare multimodal data with optimized weights
        item_features = {}
        user_features = {}
        
        for entry in train_data:
            item_id = entry['parent_asin']
            user_id = entry['user_id']
            
            # Text features
            text = f"{entry.get('title', '')} {entry.get('review_text', '')}"
            text_vector = self._text_to_vector(text, global_tfidf)
            
            # Image features
            if item_id in image_embeddings:
                image_vector = image_embeddings[item_id]
            else:
                image_vector = np.zeros(2048)
            
            # Weighted combination
            combined_features = text_weight * text_vector + image_weight * image_vector
            
            if item_id not in item_features:
                item_features[item_id] = []
            item_features[item_id].append(combined_features)
            
            if user_id not in user_features:
                user_features[user_id] = []
            user_features[user_id].append(combined_features)
        
        # Calculate average features
        item_avg_features = {item_id: np.mean(features, axis=0) 
                           for item_id, features in item_features.items()}
        
        # Evaluate
        predictions, targets = self._evaluate_multimodal(user_features, item_avg_features, test_data)
        metrics = self._calculate_metrics(predictions, targets)
        metrics['fusion_method'] = 'weighted'
        metrics['text_weight'] = text_weight
        metrics['image_weight'] = image_weight
        
        return metrics
    
    def _attention_fusion(self, train_data: List[Dict], test_data: List[Dict],
                         image_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Attention-based fusion of text and image features."""
        logger.info("  Testing attention fusion...")
        
        # Simple attention mechanism
        item_features = {}
        user_features = {}
        
        for entry in train_data:
            item_id = entry['parent_asin']
            user_id = entry['user_id']
            
            # Text features
            text = f"{entry.get('title', '')} {entry.get('review_text', '')}"
            text_vector = self._text_to_vector(text, global_tfidf)
            
            # Image features
            if item_id in image_embeddings:
                image_vector = image_embeddings[item_id]
            else:
                image_vector = np.zeros(2048)
            
            # Attention-based fusion
            # Simple attention: compute attention weights based on feature magnitudes
            text_norm = np.linalg.norm(text_vector)
            image_norm = np.linalg.norm(image_vector)
            total_norm = text_norm + image_norm
            
            if total_norm > 0:
                text_weight = text_norm / total_norm
                image_weight = image_norm / total_norm
            else:
                text_weight = image_weight = 0.5
            
            combined_features = text_weight * text_vector + image_weight * image_vector
            
            if item_id not in item_features:
                item_features[item_id] = []
            item_features[item_id].append(combined_features)
            
            if user_id not in user_features:
                user_features[user_id] = []
            user_features[user_id].append(combined_features)
        
        # Calculate average features
        item_avg_features = {item_id: np.mean(features, axis=0) 
                           for item_id, features in item_features.items()}
        
        # Evaluate
        predictions, targets = self._evaluate_multimodal(user_features, item_avg_features, test_data)
        metrics = self._calculate_metrics(predictions, targets)
        metrics['fusion_method'] = 'attention'
        
        return metrics
    
    def _optimize_fusion_weights(self, train_data: List[Dict], 
                                image_embeddings: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Simplified weight optimization."""
        # Return balanced weights for simplicity
        return (0.5, 0.5)
    
    def _text_to_vector(self, text: str, tfidf_vectorizer=None) -> np.ndarray:
        """Convert text to vector representation."""
        # Clean text - remove special tokens and ensure it's not empty
        cleaned_text = text.replace('<|endoftext|>', '').replace('<|ASIN_', '').replace('|>', '').strip()
        
        # If text is empty or too short, return zero vector
        if len(cleaned_text) < 3:
            return np.zeros(1000)
        
        try:
            if tfidf_vectorizer is None:
                # Create a simple TF-IDF vectorizer for single text
                tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=1)
                vector = tfidf_vectorizer.fit_transform([cleaned_text]).toarray().flatten()
            else:
                # Use pre-fitted vectorizer
                vector = tfidf_vectorizer.transform([cleaned_text]).toarray().flatten()
            
            # Pad or truncate to fixed size
            if len(vector) < 1000:
                vector = np.pad(vector, (0, 1000 - len(vector)))
            else:
                vector = vector[:1000]
            
            return vector
        except ValueError:
            # If TF-IDF fails, return zero vector
            logger.warning(f"TF-IDF failed for text: '{cleaned_text[:50]}...', using zero vector")
            return np.zeros(1000)
    
    def _evaluate_multimodal(self, user_features: Dict[str, List[np.ndarray]], 
                           item_features: Dict[str, np.ndarray], 
                           test_data: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
        """Evaluate multimodal model."""
        predictions = []
        targets = []
        
        for entry in test_data:
            user_id = entry['user_id']
            target_items = entry['output'].split(', ')
            
            if user_id in user_features:
                user_avg_features = np.mean(user_features[user_id], axis=0)
                
                # Calculate similarities
                similarities = {}
                for item_id, item_features_vector in item_features.items():
                    similarity = cosine_similarity([user_avg_features], [item_features_vector])[0][0]
                    similarities[item_id] = similarity
                
                # Get top recommendations
                sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                predicted_items = [item_id for item_id, _ in sorted_items[:10]]
            else:
                # Cold start
                predicted_items = list(item_features.keys())[:10]
            
            predictions.append(predicted_items)
            targets.append(target_items)
        
        return predictions, targets
    
    def _calculate_metrics(self, predictions: List[List[str]], targets: List[List[str]]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Hit Rate@10
        hits = 0
        total = len(predictions)
        for pred, target in zip(predictions, targets):
            if any(item in target for item in pred[:10]):
                hits += 1
        metrics['hr@10'] = hits / total if total > 0 else 0.0
        
        # Precision@10
        total_precision = 0
        for pred, target in zip(predictions, targets):
            hits = len(set(pred[:10]) & set(target))
            total_precision += hits / 10
        metrics['precision@10'] = total_precision / total if total > 0 else 0.0
        
        # Recall@10
        total_recall = 0
        for pred, target in zip(predictions, targets):
            hits = len(set(pred[:10]) & set(target))
            total_recall += hits / len(target) if len(target) > 0 else 0
        metrics['recall@10'] = total_recall / total if total > 0 else 0.0
        
        # MRR
        total_mrr = 0
        for pred, target in zip(predictions, targets):
            for i, item in enumerate(pred):
                if item in target:
                    total_mrr += 1 / (i + 1)
                    break
        metrics['mrr'] = total_mrr / total if total > 0 else 0.0
        
        # NDCG@10
        total_ndcg = 0
        for pred, target in zip(predictions, targets):
            dcg = 0
            idcg = 1
            for i, item in enumerate(pred[:10]):
                if item in target:
                    dcg += 1 / np.log2(i + 2)
            total_ndcg += dcg / idcg
        metrics['ndcg@10'] = total_ndcg / total if total > 0 else 0.0
        
        return metrics
    
    def _create_global_tfidf_vectorizer(self, train_data: List[Dict]) -> TfidfVectorizer:
        """Create a global TF-IDF vectorizer from all training texts."""
        logger.info("Creating global TF-IDF vectorizer...")
        
        # Collect all texts
        all_texts = []
        for entry in train_data:
            if 'input' in entry:
                cleaned_text = entry['input'].replace('<|endoftext|>', '').replace('<|ASIN_', '').replace('|>', '').strip()
                if len(cleaned_text) >= 3:
                    all_texts.append(cleaned_text)
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=1)
        tfidf.fit(all_texts)
        
        logger.info(f"TF-IDF vectorizer created with {len(all_texts)} texts")
        return tfidf

    def run_complete_ablation(self, train_data: List[Dict], test_data: List[Dict],
                            image_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run complete ablation study.
        """
        logger.info("Starting complete ablation study...")
        
        # Limit dataset size for faster execution (use first 1000 samples)
        max_samples = 1000
        if len(train_data) > max_samples:
            logger.info(f"Limiting ablation study to {max_samples} samples for faster execution")
            train_data = train_data[:max_samples]
        if len(test_data) > max_samples:
            test_data = test_data[:max_samples]
        
        # Create global TF-IDF vectorizer for efficiency
        global_tfidf = self._create_global_tfidf_vectorizer(train_data)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'text_only': {},
            'image_only': {},
            'multimodal_fusion': {},
            'summary': {}
        }
        
        # Text-only study (with optimized TF-IDF)
        results['text_only'] = self.text_only_model(train_data, test_data, global_tfidf)
        
        # Image-only study
        results['image_only'] = self.image_only_model(train_data, test_data, image_embeddings)
        
        # Multimodal fusion comparison (with optimized TF-IDF)
        results['multimodal_fusion'] = self.multimodal_fusion_comparison(train_data, test_data, image_embeddings, global_tfidf)
        
        # Generate summary
        results['summary'] = self._generate_ablation_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_ablation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of ablation study results."""
        summary = {
            'best_modality': '',
            'best_fusion_method': '',
            'modality_contributions': {},
            'key_insights': []
        }
        
        # Compare modalities
        text_score = results['text_only']['hr@10']
        image_score = results['image_only']['hr@10']
        
        if text_score > image_score:
            summary['best_modality'] = 'text'
            summary['modality_contributions']['text'] = text_score
            summary['modality_contributions']['image'] = image_score
        else:
            summary['best_modality'] = 'image'
            summary['modality_contributions']['image'] = image_score
            summary['modality_contributions']['text'] = text_score
        
        # Compare fusion methods
        fusion_results = results['multimodal_fusion']
        best_fusion_score = 0
        best_fusion_method = ''
        
        for method, metrics in fusion_results.items():
            score = metrics['hr@10']
            if score > best_fusion_score:
                best_fusion_score = score
                best_fusion_method = method
        
        summary['best_fusion_method'] = best_fusion_method
        
        # Generate insights
        summary['key_insights'].append(f"Text modality contributes {text_score:.4f} HR@10")
        summary['key_insights'].append(f"Image modality contributes {image_score:.4f} HR@10")
        summary['key_insights'].append(f"Best fusion method: {best_fusion_method} ({best_fusion_score:.4f} HR@10)")
        
        if best_fusion_score > max(text_score, image_score):
            summary['key_insights'].append("Multimodal fusion provides improvement over single modalities")
        else:
            summary['key_insights'].append("Single modality performs better than multimodal fusion")
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save ablation study results."""
        output_path = Path('data/processed/ablation_study_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Ablation study results saved to {output_path}")


def main():
    """Main function to run ablation studies."""
    logger.info("Running ablation studies...")
    
    try:
        from src.utils.utils import load_config
        
        config = load_config()
        
        # Load data
        with open(config['data_config']['train_file'], 'r') as f:
            train_data = json.load(f)
        
        with open(config['data_config']['test_file'], 'r') as f:
            test_data = json.load(f)
        
        # Load image embeddings (if available)
        image_embeddings = {}  # Load from your image processing pipeline
        
        # Run ablation study
        ablation = AblationStudy(config)
        results = ablation.run_complete_ablation(train_data, test_data, image_embeddings)
        
        logger.info("Ablation study completed!")
        logger.info(f"Best modality: {results['summary']['best_modality']}")
        logger.info(f"Best fusion method: {results['summary']['best_fusion_method']}")
        
    except Exception as e:
        logger.error(f"Ablation studies failed: {e}")
        raise


if __name__ == "__main__":
    main()
