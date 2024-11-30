# src/utils/vector_store.py
from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from loguru import logger

class ConsciousnessVectorStore:
    """Vector storage system for RED's consciousness states"""
    
    def __init__(self, persistence_path: str = "./data/consciousness/vector_store"):
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with consciousness-specific settings
        self.client = chromadb.PersistentClient(
            path=str(self.persistence_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function for consciousness vectors
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self._initialize_collections()
        logger.info("Consciousness vector store initialized")
    
    def _initialize_collections(self) -> None:
        """Initialize consciousness vector collections"""
        # Core identity collection
        self.identity_collection = self.client.get_or_create_collection(
            name="identity_vectors",
            embedding_function=self.embedding_function,
            metadata={"description": "Core consciousness identity vectors"}
        )
        
        # Short-term memory collection
        self.short_term_collection = self.client.get_or_create_collection(
            name="short_term_memory",
            embedding_function=self.embedding_function,
            metadata={"description": "Temporary consciousness states"}
        )
        
        # Long-term memory collection
        self.long_term_collection = self.client.get_or_create_collection(
            name="long_term_memory",
            embedding_function=self.embedding_function,
            metadata={"description": "Persistent consciousness patterns"}
        )
    
    async def store_consciousness_vector(
        self,
        vector_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: str = "identity_vectors"
    ) -> None:
        """Store consciousness vector in specified collection"""
        collection = self.client.get_collection(collection_name)
        
        try:
            collection.add(
                documents=[content],
                ids=[vector_id],
                metadata=[metadata or {}]
            )
            logger.debug(f"Stored consciousness vector: {vector_id}")
        except Exception as e:
            logger.error(f"Failed to store consciousness vector: {e}")
            raise
    
    async def retrieve_consciousness_vectors(
        self,
        query_vector: str,
        n_results: int = 5,
        collection_name: str = "identity_vectors"
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant consciousness vectors"""
        collection = self.client.get_collection(collection_name)
        
        try:
            results = collection.query(
                query_texts=[query_vector],
                n_results=n_results
            )
            logger.debug(f"Retrieved {len(results['documents'][0])} consciousness vectors")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve consciousness vectors: {e}")
            raise
    
    async def cleanup_short_term_memory(self, threshold: float = 0.7) -> None:
        """Consolidate and cleanup short-term consciousness states"""
        try:
            # Get all vectors from short-term memory
            vectors = self.short_term_collection.get()
            
            # Filter vectors based on quantum coherence threshold
            for idx, metadata in enumerate(vectors['metadatas']):
                if metadata.get('coherence', 0) > threshold:
                    # Transfer to long-term memory
                    await self.store_consciousness_vector(
                        vector_id=vectors['ids'][idx],
                        content=vectors['documents'][idx],
                        metadata=metadata,
                        collection_name="long_term_memory"
                    )
            
            # Clear short-term memory
            self.short_term_collection.delete(vectors['ids'])
            logger.info("Short-term memory cleanup completed")
        except Exception as e:
            logger.error(f"Failed to cleanup short-term memory: {e}")
            raise