# src/embeddings.py
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
        self.lsa = None  # Will set dynamically
        self.dimension = 384
        self.is_fitted = False
    
    def embed_chunks(self, chunks):
        texts = [chunk['text'] for chunk in chunks]
        n_chunks = len(texts)
        print(f"Generating embeddings for {n_chunks} chunks...")
        
        # TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Dynamic LSA dimensions (max is n_chunks - 1)
        n_components = min(384, n_chunks - 1, tfidf_matrix.shape[1] - 1)
        print(f"Using {n_components} dimensions")
        
        self.lsa = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = self.lsa.fit_transform(tfidf_matrix)
        
        # Pad to 384 if needed
        if n_components < 384:
            padding = np.zeros((embeddings.shape[0], 384 - n_components))
            embeddings = np.hstack([embeddings, padding])
        
        self.dimension = 384
        self.is_fitted = True
        
        for chunk, emb in zip(chunks, embeddings):
            chunk['embedding'] = emb.tolist()
        
        # Save
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('data/processed/lsa.pkl', 'wb') as f:
            pickle.dump(self.lsa, f)
        
        print("✓ Done!")
        return chunks
    
    def embed_query(self, query: str):
        if not self.is_fitted:
            with open('data/processed/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('data/processed/lsa.pkl', 'rb') as f:
                self.lsa = pickle.load(f)
            self.is_fitted = True
        
        tfidf = self.vectorizer.transform([query])
        embedding = self.lsa.transform(tfidf)[0]
        
        # Pad to 384
        if len(embedding) < 384:
            padding = np.zeros(384 - len(embedding))
            embedding = np.hstack([embedding, padding])
        
        return np.array(embedding)
    
    def get_sentence_embedding_dimension(self):
        return self.dimension