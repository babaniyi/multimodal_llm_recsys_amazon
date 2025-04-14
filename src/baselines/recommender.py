import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    def __init__(self, n_factors: int = 100):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
    
    def fit(self, interactions_df: pd.DataFrame):
        """
        Fit the collaborative filtering model.
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        # Create user and item mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create sparse matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        ratings = csr_matrix((
            interactions_df['rating'].values,
            (
                [self.user_mapping[user] for user in interactions_df['user_id']],
                [self.item_mapping[item] for item in interactions_df['item_id']]
            )
        ), shape=(n_users, n_items))
        
        # Perform matrix factorization
        U, S, Vt = np.linalg.svd(ratings.toarray(), full_matrices=False)
        self.user_factors = U[:, :self.n_factors]
        self.item_factors = Vt[:self.n_factors, :].T
    
    def predict(self, user_id: int, n_items: int = 10) -> List[int]:
        """
        Predict top-n items for a user.
        
        Args:
            user_id: User ID
            n_items: Number of items to recommend
            
        Returns:
            List of recommended item IDs
        """
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(self.item_factors, user_vector)
        
        # Get top-n items
        top_indices = np.argsort(scores)[-n_items:][::-1]
        return [self.reverse_item_mapping[idx] for idx in top_indices]

class ContentBasedRecommender:
    def __init__(self, image_embeddings: Dict[int, np.ndarray] = None):
        self.image_embeddings = image_embeddings
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.item_vectors = None
        self.item_ids = None
    
    def fit(self, items_df: pd.DataFrame):
        """
        Fit the content-based model.
        
        Args:
            items_df: DataFrame with columns ['item_id', 'title', 'description']
        """
        # Create text features
        text_features = items_df['title'] + ' ' + items_df['description']
        self.item_vectors = self.tfidf.fit_transform(text_features)
        self.item_ids = items_df['item_id'].values
    
    def predict(self, item_id: int, n_items: int = 10) -> List[int]:
        """
        Predict similar items based on content.
        
        Args:
            item_id: Item ID
            n_items: Number of items to recommend
            
        Returns:
            List of recommended item IDs
        """
        # Find item index
        item_idx = np.where(self.item_ids == item_id)[0][0]
        
        # Calculate similarities
        if self.image_embeddings is not None:
            # Use both text and image features
            text_similarities = cosine_similarity(
                self.item_vectors[item_idx:item_idx+1],
                self.item_vectors
            ).flatten()
            
            image_similarities = cosine_similarity(
                self.image_embeddings[item_id:item_id+1],
                list(self.image_embeddings.values())
            ).flatten()
            
            # Combine similarities (weighted average)
            similarities = 0.5 * text_similarities + 0.5 * image_similarities
        else:
            # Use only text features
            similarities = cosine_similarity(
                self.item_vectors[item_idx:item_idx+1],
                self.item_vectors
            ).flatten()
        
        # Get top-n items (excluding the input item)
        top_indices = np.argsort(similarities)[-n_items-1:-1][::-1]
        return [self.item_ids[idx] for idx in top_indices]

class HybridRecommender:
    def __init__(
        self,
        cf_weight: float = 0.5,
        content_weight: float = 0.5,
        n_factors: int = 100
    ):
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.cf_model = CollaborativeFiltering(n_factors)
        self.content_model = ContentBasedRecommender()
    
    def fit(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        image_embeddings: Dict[int, np.ndarray] = None
    ):
        """
        Fit both models.
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating']
            items_df: DataFrame with columns ['item_id', 'title', 'description']
            image_embeddings: Dictionary mapping item_id to image embeddings
        """
        self.cf_model.fit(interactions_df)
        self.content_model.image_embeddings = image_embeddings
        self.content_model.fit(items_df)
    
    def predict(
        self,
        user_id: int,
        item_id: int,
        n_items: int = 10
    ) -> List[int]:
        """
        Predict items using both models.
        
        Args:
            user_id: User ID
            item_id: Item ID
            n_items: Number of items to recommend
            
        Returns:
            List of recommended item IDs
        """
        # Get predictions from both models
        cf_predictions = self.cf_model.predict(user_id, n_items)
        content_predictions = self.content_model.predict(item_id, n_items)
        
        # Combine predictions (weighted voting)
        item_scores = {}
        
        for rank, item in enumerate(cf_predictions):
            item_scores[item] = item_scores.get(item, 0) + self.cf_weight * (1 / (rank + 1))
        
        for rank, item in enumerate(content_predictions):
            item_scores[item] = item_scores.get(item, 0) + self.content_weight * (1 / (rank + 1))
        
        # Sort by combined score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:n_items]]

class PopularityRecommender:
    def __init__(self):
        self.popular_items = None
        self.item_counts = None
    
    def fit(self, interactions_df: pd.DataFrame):
        """
        Fit the popularity-based model.
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        # Count interactions per item
        self.item_counts = interactions_df.groupby('item_id').size()
        # Sort items by popularity
        self.popular_items = self.item_counts.sort_values(ascending=False).index.tolist()
    
    def predict(self, n_items: int = 10) -> List[int]:
        """
        Predict top-n most popular items.
        
        Args:
            n_items: Number of items to recommend
            
        Returns:
            List of recommended item IDs
        """
        return self.popular_items[:n_items] 