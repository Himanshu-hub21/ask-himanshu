from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    """
    Manage Qdrant vector database
    """
    
    def __init__(self):
        # Connect to Qdrant Cloud (FREE tier)
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        self.collection_name = "himanshu_knowledge"
        self.vector_size = 384  # for all-MiniLM-L6-v2
    
    def create_collection(self):
        """
        Create collection if doesn't exist
        """
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists")
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Collection '{self.collection_name}' created")
    
    def insert_chunks(self, chunks: List[Dict]):
        """
        Insert chunks with embeddings into Qdrant
        """
        points = []
        
        for idx, chunk in enumerate(chunks):
            point = PointStruct(
                id=idx,
                vector=chunk['embedding'],
                payload={
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'chunk_id': chunk['chunk_id'],
                    'has_pii': chunk['has_pii'],
                    'word_count': chunk['word_count']
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        
        print(f"✓ Inserted {len(points)} chunks into Qdrant")
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        Search for similar chunks
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Extract payload
        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append({
                'text': result.payload['text'],
                'source': result.payload['source'],
                'page': result.payload['page'],
                'score': result.score,
                'has_pii': result.payload.get('has_pii', False)
            })
        
        return retrieved_chunks