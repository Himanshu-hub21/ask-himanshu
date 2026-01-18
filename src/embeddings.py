import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import os

class EmbeddingGenerator:
    def __init__(self, model_name: str = "tfidf-lsa"):
        print("Using TF-IDF + LSA embeddings")
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            ngram_range=(1,2),
            min_df=1
        )
        self.lsa = None
        self.dimension = 384
        self.is_fitted = False
    
    def embed_chunks(self, chunks):
        texts = [chunk['text'] for chunk in chunks]
        n_chunks = len(texts)
        print(f"Generating embeddings for {n_chunks} chunks...")
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_components = min(384, n_chunks - 1, tfidf_matrix.shape[1] - 1)
        
        self.lsa = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = self.lsa.fit_transform(tfidf_matrix)
        
        if n_components < 384:
            padding = np.zeros((embeddings.shape[0], 384 - n_components))
            embeddings = np.hstack([embeddings, padding])
        
        self.dimension = 384
        self.is_fitted = True
        
        for chunk, emb in zip(chunks, embeddings):
            chunk['embedding'] = emb.tolist()
        
        print("✓ Done!")
        return chunks
    
    def embed_query(self, query: str):
        # For deployment: use pre-trained from Qdrant
        # Simple fallback embedding
        words = query.lower().split()
        embedding = np.zeros(384)
        
        # Simple hash-based embedding
        for i, word in enumerate(words[:20]):
            hash_val = hash(word) % 384
            embedding[hash_val] += 1.0 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_sentence_embedding_dimension(self):
        return self.dimension