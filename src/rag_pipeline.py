from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler
from src.privacy_filter import PrivacyFilter
from typing import Dict

class RAGPipeline:
    """
    Complete RAG pipeline for answering queries
    """
    
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.llm = LLMHandler()
        self.privacy = PrivacyFilter()
    
    def answer_query(self, query: str, top_k: int = 3) -> Dict:
        """
        Complete pipeline: query → retrieve → generate
        """
        # Step 1: Check for PII request
        if self.privacy.is_pii_request(query):
            return {
                'answer': self.privacy.handle_pii_request(query),
                'sources': [],
                'is_pii_response': True
            }
        
        # Step 2: Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Step 3: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k
        )
        
        # Step 4: Build prompt with context
        context = "\n\n".join([
            f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        prompt = f"""Context information from Himanshu's documents:

        {context}

        Question: {query}

        Based ONLY on the context above, provide a clear and accurate answer. If the information is not in the context, say "I don't have that information in the documents provided."

        Answer:"""
        
        # Step 5: Generate response
        answer = self.llm.generate_response(prompt)
        
        # Step 6: Redact any leaked PII
        answer = self.privacy.redact_pii_from_text(answer)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'score': chunk['score']
                }
                for chunk in retrieved_chunks
            ],
            'is_pii_response': False
        }
