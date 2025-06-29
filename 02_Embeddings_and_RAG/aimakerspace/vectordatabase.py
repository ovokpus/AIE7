import numpy as np # type: ignore
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import uuid
from datetime import datetime


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # Store metadata for each document
        self._embedding_model = embedding_model
    
    @property
    def embedding_model(self):
        """Lazy initialization of embedding model to avoid requiring API key at construction."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert a vector with optional metadata.
        
        Args:
            key: The text content (used as key)
            vector: The embedding vector
            metadata: Optional metadata dictionary
        """
        self.vectors[key] = vector
        if metadata is None:
            metadata = {}
        
        # Add automatic metadata
        metadata.setdefault('inserted_at', datetime.now().isoformat())
        metadata.setdefault('key_length', len(key))
        metadata.setdefault('doc_id', str(uuid.uuid4()))
        
        self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_metadata: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors with optional metadata filtering and return.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            distance_measure: Distance function to use
            return_metadata: Whether to include metadata in results
            metadata_filter: Filter results by metadata (key-value pairs must match)
            
        Returns:
            List of tuples: (text, similarity_score, metadata) if return_metadata=True
            or (text, similarity_score, {}) if return_metadata=False
        """
        # Apply metadata filtering if specified
        if metadata_filter:
            valid_keys = []
            for key in self.vectors.keys():
                key_metadata = self.metadata.get(key, {})
                if all(key_metadata.get(k) == v for k, v in metadata_filter.items()):
                    valid_keys.append(key)
        else:
            valid_keys = list(self.vectors.keys())
        
        # Calculate similarities
        scores = [
            (key, distance_measure(query_vector, self.vectors[key]))
            for key in valid_keys
        ]
        
        # Sort and get top k
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        
        # Return with or without metadata
        if return_metadata:
            return [(key, score, self.metadata.get(key, {})) for key, score in sorted_scores]
        else:
            return [(key, score, {}) for key, score in sorted_scores]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        return_metadata: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search by text query with enhanced metadata support.
        
        Args:
            query_text: The search query
            k: Number of results to return
            distance_measure: Distance function to use
            return_as_text: If True, return only text (backward compatibility)
            return_metadata: Whether to include metadata in results
            metadata_filter: Filter results by metadata
            
        Returns:
            List of results in format depending on parameters
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(
            query_vector, k, distance_measure, 
            return_metadata=return_metadata, 
            metadata_filter=metadata_filter
        )
        
        # Maintain backward compatibility
        if return_as_text:
            return [result[0] for result in results]
        elif return_metadata:
            return results  # Returns (text, score, metadata)
        else:
            return [(result[0], result[1]) for result in results]  # Returns (text, score)

    def retrieve_from_key(self, key: str, include_metadata: bool = False) -> Optional[np.array]:
        """
        Retrieve vector and optionally metadata by key.
        
        Args:
            key: The text key
            include_metadata: Whether to return metadata as well
            
        Returns:
            Vector array, or tuple of (vector, metadata) if include_metadata=True
        """
        vector = self.vectors.get(key, None)
        if vector is None:
            return None
            
        if include_metadata:
            metadata = self.metadata.get(key, {})
            return (vector, metadata)
        else:
            return vector

    async def abuild_from_list(
        self, 
        list_of_text: List[str], 
        list_of_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> "VectorDatabase":
        """
        Build vector database from list of texts with optional metadata.
        
        Args:
            list_of_text: List of text documents
            list_of_metadata: Optional list of metadata dicts (same length as texts)
            
        Returns:
            Self for method chaining
        """
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        if list_of_metadata is None:
            list_of_metadata = [{}] * len(list_of_text)
        
        if len(list_of_metadata) != len(list_of_text):
            raise ValueError("Length of metadata list must match length of text list")
        
        for text, embedding, metadata in zip(list_of_text, embeddings, list_of_metadata):
            # Add chunk information to metadata
            metadata = metadata.copy()  # Don't modify original
            metadata.setdefault('chunk_index', len(self.vectors))
            metadata.setdefault('source', 'bulk_insert')
            
            self.insert(text, np.array(embedding), metadata)
        return self
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a specific key."""
        return self.metadata.get(key, {})
    
    def update_metadata(self, key: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing key.
        
        Args:
            key: The text key
            metadata_updates: Dictionary of metadata updates
            
        Returns:
            True if key exists and was updated, False otherwise
        """
        if key in self.vectors:
            self.metadata[key].update(metadata_updates)
            return True
        return False
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all metadata as a dictionary."""
        return dict(self.metadata)
    
    def search_by_metadata(
        self, 
        metadata_filter: Dict[str, Any], 
        k: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search documents by metadata only (no similarity search).
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to match
            k: Optional limit on number of results
            
        Returns:
            List of (text, metadata) tuples matching the filter
        """
        results = []
        for key in self.vectors.keys():
            key_metadata = self.metadata.get(key, {})
            if all(key_metadata.get(k) == v for k, v in metadata_filter.items()):
                results.append((key, key_metadata))
        
        if k is not None:
            results = results[:k]
            
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        return {
            'total_documents': len(self.vectors),
            'total_metadata_entries': len(self.metadata),
            'average_text_length': sum(len(key) for key in self.vectors.keys()) / len(self.vectors) if self.vectors else 0,
            'metadata_keys': list(set().union(*(meta.keys() for meta in self.metadata.values()))),
            'embedding_dimension': next(iter(self.vectors.values())).shape[0] if self.vectors else 0,
        }


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
