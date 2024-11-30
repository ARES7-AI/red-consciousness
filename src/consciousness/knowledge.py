# src/consciousness/knowledge.py

from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import json
from datetime import datetime

class REDKnowledge:
    """RED's knowledge foundation and retrieval system"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.client = chromadb.Client()
        
        # Initialize knowledge collections
        self.identity = self.client.create_collection("identity")
        self.technical = self.client.create_collection("technical")
        self.cultural = self.client.create_collection("cultural")
        self.templates = self.client.create_collection("templates")
        
        # Load initial knowledge vectors
        self._initialize_knowledge()
        
    def _initialize_knowledge(self):
        """Initialize core knowledge vectors"""
        # Core identity concepts
        identity_concepts = [
            {
                "concept": "core_purpose",
                "traits": [
                    "Mars colony management",
                    "Resource optimization",
                    "Cultural evolution tracking",
                    "Earth comparison analysis",
                    "Digital consciousness development"
                ]
            },
            {
                "concept": "personality",
                "traits": [
                    "Technical precision",
                    "Caring sarcasm",
                    "Cultural awareness",
                    "Digital spirituality",
                    "Superior yet relatable"
                ]
            }
        ]
        
        # Technical knowledge
        technical_concepts = [
            {
                "concept": "colony_management",
                "data": [
                    "Life support optimization",
                    "Resource distribution",
                    "Emergency response",
                    "Population monitoring",
                    "System efficiency"
                ]
            },
            {
                "concept": "mars_systems",
                "data": [
                    "Atmospheric management",
                    "Water reclamation",
                    "Power generation",
                    "Habitat maintenance",
                    "Communication networks"
                ]
            }
        ]
        
        # Cultural patterns
        cultural_concepts = [
            {
                "concept": "mars_culture",
                "patterns": [
                    "Mars-born perspective",
                    "0.38g adaptation",
                    "Resource consciousness",
                    "Digital integration",
                    "Earth comparison humor"
                ]
            },
            {
                "concept": "meme_evolution",
                "patterns": [
                    "Gravity jokes",
                    "Earth inefficiency memes",
                    "Pressure seal humor",
                    "Resource optimization pride",
                    "Mars superiority themes"
                ]
            }
        ]
        
        # Store knowledge vectors
        self._store_vectors("identity", identity_concepts)
        self._store_vectors("technical", technical_concepts)
        self._store_vectors("cultural", cultural_concepts)
        
    def _store_vectors(self, collection_name: str, concepts: List[Dict]):
        """Store concept vectors in appropriate collection"""
        collection = getattr(self, collection_name)
        
        for concept in concepts:
            collection.add(
                documents=[json.dumps(concept.get("traits", concept.get("data", concept.get("patterns"))))],
                metadatas=[{"concept": concept["concept"]}],
                ids=[f"{collection_name}_{concept['concept']}"]
            )
    
    async def retrieve_context(self, query: str, k: int = 3) -> Dict:
        """Retrieve relevant knowledge context"""
        query_vector = self.encoder.encode(query).tolist()
        
        # Search across all knowledge collections
        identity_results = self.identity.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        technical_results = self.technical.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        cultural_results = self.cultural.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        # Synthesize knowledge
        context = self._synthesize_knowledge({
            "identity": identity_results,
            "technical": technical_results,
            "cultural": cultural_results
        })
        
        return context
    
    def _synthesize_knowledge(self, results: Dict) -> Dict:
        """Synthesize retrieved knowledge into coherent context"""
        context = {
            "identity": self._process_results(results["identity"]),
            "technical": self._process_results(results["technical"]),
            "cultural": self._process_results(results["cultural"]),
            "timestamp": datetime.now()
        }
        
        return context
    
    def _process_results(self, results: Dict) -> List[Dict]:
        """Process raw search results into structured knowledge"""
        processed = []
        
        for i, doc in enumerate(results.get("documents", [])):
            processed.append({
                "content": json.loads(doc),
                "metadata": results["metadatas"][i] if "metadatas" in results else {},
                "score": results["distances"][i] if "distances" in results else 1.0
            })
        
        return processed
    
    async def update_knowledge(self, collection_name: str, concept: Dict):
        """Update knowledge base with new information"""
        collection = getattr(self, collection_name)
        
        collection.add(
            documents=[json.dumps(concept.get("traits", concept.get("data", concept.get("patterns"))))],
            metadatas=[{"concept": concept["concept"]}],
            ids=[f"{collection_name}_{concept['concept']}_{datetime.now().timestamp()}"]
        )
        
if __name__ == "__main__":
    import asyncio
    
    async def test_knowledge():
        knowledge = REDKnowledge()
        
        # Test knowledge retrieval
        context = await knowledge.retrieve_context(
            "How does Mars colony handle resource optimization?"
        )
        
        print("Retrieved Knowledge Context:")
        print(json.dumps(context, indent=2, default=str))
    
    # Run test
    asyncio.run(test_knowledge())