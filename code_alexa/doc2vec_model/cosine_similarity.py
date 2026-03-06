"""
Cosine similarity computation for vectors
"""
import numpy as np


def dot_product2(v1, v2):
    """
    Compute dot product between two vectors
    
    Args:
        v1: First vector (numpy array or list)
        v2: Second vector (numpy array or list)
        
    Returns:
        Dot product (scalar)
    """
    return np.dot(v1, v2)


def vector_cos5(v1, v2):
    """
    Compute cosine similarity between two vectors
    
    Args:
        v1: First vector (numpy array or list)
        v2: Second vector (numpy array or list)
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute cosine similarity: (v1 · v2) / (||v1|| * ||v2||)
    dot_prod = dot_product2(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_prod / (norm_v1 * norm_v2)


def cosine_similarity(v1, v2):
    """
    Alias for vector_cos5 with more intuitive name
    
    Args:
        v1: First vector (numpy array or list)
        v2: Second vector (numpy array or list)
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    return vector_cos5(v1, v2)
