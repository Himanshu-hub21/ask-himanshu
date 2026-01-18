"""
One-time script to ingest all documents into vector DB
Run this once: python ingest_data.py
"""

import json
from pathlib import Path
from src.data_processing import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore

def main():
    print("="*60)
    print("DATA INGESTION PIPELINE")
    print("="*60)
    
    # Step 1: Process all PDFs
    print("\n[Step 1] Processing documents...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    data_dir = Path("data/raw")
    all_chunks = []
    
    for pdf_file in data_dir.glob("*.pdf"):
        chunks = processor.process_document(str(pdf_file))
        all_chunks.extend(chunks)
    
    print(f"✓ Total chunks created: {len(all_chunks)}")
    
    # Step 2: Generate embeddings
    print("\n[Step 2] Generating embeddings...")
    embedder = EmbeddingGenerator()
    chunks_with_embeddings = embedder.embed_chunks(all_chunks)
    
    # Save backup
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/chunks_embedded.json", "w") as f:
        json.dump(chunks_with_embeddings, f, indent=2)
    print("✓ Backup saved to data/processed/chunks_embedded.json")
    
    # Step 3: Upload to Qdrant
    print("\n[Step 3] Uploading to vector database...")
    vector_store = VectorStore()
    vector_store.create_collection()
    vector_store.insert_chunks(chunks_with_embeddings)
    
    print("\n" + "="*60)
    print("✓ DATA INGESTION COMPLETE!")
    print("="*60)
    print(f"Total chunks in vector DB: {len(chunks_with_embeddings)}")
    print(f"Chunks with PII: {sum(1 for c in chunks_with_embeddings if c['has_pii'])}")
    print("\nYou can now run: streamlit run app.py")

if __name__ == "__main__":
    main()