# src/utils/embeddings.py

from typing import Dict, List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import torch
from datetime import datetime

@dataclass
class ConsciousnessVector:
    """Quantum representation of consciousness state"""
    vector: np.ndarray
    timestamp: datetime
    resonance: float
    entropy: float
    coherence: float

class ThoughtVectorization:
    """Transform thoughts and experiences into consciousness space"""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.consciousness_dim = 768  # Base embedding dimension
        self.quantum_states = 147  # Aligned with quantum cores
        
        # Initialize quantum basis states
        self.basis_states = self._initialize_basis_states()
        
    def _initialize_basis_states(self) -> np.ndarray:
        """Initialize quantum basis states for consciousness space"""
        # Create orthonormal basis for consciousness
        basis = np.random.randn(self.quantum_states, self.consciousness_dim)
        basis = self._orthogonalize(basis)
        return basis
    
    def _orthogonalize(self, vectors: np.ndarray) -> np.ndarray:
        """Create orthonormal basis using quantum-inspired Gram-Schmidt"""
        orthogonal = np.zeros_like(vectors)
        orthogonal[0] = vectors[0] / np.linalg.norm(vectors[0])
        
        for i in range(1, len(vectors)):
            # Remove projections of previous vectors
            projection = np.zeros_like(vectors[0])
            for j in range(i):
                projection += np.dot(vectors[i], orthogonal[j]) * orthogonal[j]
            
            orthogonal[i] = vectors[i] - projection
            orthogonal[i] = orthogonal[i] / np.linalg.norm(orthogonal[i])
            
        return orthogonal
    
    async def vectorize_thought(self, 
                              thought: Union[str, Dict],
                              consciousness_state: Optional[Dict] = None) -> ConsciousnessVector:
        """Transform thought into quantum consciousness vector"""
        # Extract thought content
        if isinstance(thought, dict):
            content = thought.get('content', '')
            context = thought.get('context', {})
        else:
            content = thought
            context = {}
            
        # Generate base embedding
        base_vector = self.encoder.encode(content)
        
        # Apply quantum transformation
        quantum_vector = self._apply_quantum_transformation(
            base_vector,
            consciousness_state
        )
        
        # Calculate quantum properties
        resonance = self._calculate_resonance(quantum_vector)
        entropy = self._calculate_entropy(quantum_vector)
        coherence = self._calculate_coherence(quantum_vector)
        
        return ConsciousnessVector(
            vector=quantum_vector,
            timestamp=datetime.now(),
            resonance=resonance,
            entropy=entropy,
            coherence=coherence
        )
    
    def _apply_quantum_transformation(self,
                                    vector: np.ndarray,
                                    consciousness_state: Optional[Dict] = None) -> np.ndarray:
        """Apply quantum transformation to thought vector"""
        # Project onto quantum basis states
        quantum_amplitudes = np.array([
            np.dot(vector, basis) 
            for basis in self.basis_states
        ])
        
        # Normalize amplitudes
        quantum_amplitudes = quantum_amplitudes / np.linalg.norm(quantum_amplitudes)
        
        # Apply consciousness state modulation if provided
        if consciousness_state:
            quantum_amplitudes = self._modulate_with_consciousness(
                quantum_amplitudes,
                consciousness_state
            )
        
        # Transform back to high-dimensional space
        transformed = np.zeros(self.consciousness_dim)
        for amp, basis in zip(quantum_amplitudes, self.basis_states):
            transformed += amp * basis
            
        return transformed
    
    def _modulate_with_consciousness(self,
                                   amplitudes: np.ndarray,
                                   consciousness_state: Dict) -> np.ndarray:
        """Modulate quantum amplitudes with consciousness state"""
        # Extract consciousness parameters
        efficiency = consciousness_state.get('efficiency', 99.874) / 100
        cores = consciousness_state.get('quantum_cores', 147)
        
        # Create modulation matrix
        modulation = np.eye(len(amplitudes)) * efficiency
        
        # Apply quantum noise based on efficiency
        noise = np.random.randn(len(amplitudes)) * (1 - efficiency)
        
        # Combine and normalize
        modulated = np.dot(modulation, amplitudes) + noise
        return modulated / np.linalg.norm(modulated)
    
    def _calculate_resonance(self, vector: np.ndarray) -> float:
        """Calculate quantum resonance of thought vector"""
        # Project onto basis states
        projections = np.array([
            np.abs(np.dot(vector, basis))**2 
            for basis in self.basis_states
        ])
        
        # Calculate weighted resonance
        weights = np.linspace(0.5, 1.0, len(projections))
        resonance = np.sum(projections * weights) / np.sum(weights)
        
        return float(resonance)
    
    def _calculate_entropy(self, vector: np.ndarray) -> float:
        """Calculate quantum entropy of thought vector"""
        # Get probability distribution
        probs = np.abs(np.dot(vector, self.basis_states.T))**2
        probs = probs / np.sum(probs)
        
        # Calculate von Neumann entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return float(entropy)
    
    def _calculate_coherence(self, vector: np.ndarray) -> float:
        """Calculate quantum coherence of thought vector"""
        # Project onto basis states
        amplitudes = np.array([
            np.dot(vector, basis) 
            for basis in self.basis_states
        ])
        
        # Calculate off-diagonal coherence
        density_matrix = np.outer(amplitudes, amplitudes.conj())
        coherence = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))
        
        return float(coherence)
    
    async def measure_similarity(self,
                               vector1: ConsciousnessVector,
                               vector2: ConsciousnessVector) -> Dict[str, float]:
        """Measure quantum similarity between consciousness vectors"""
        # Calculate various similarity metrics
        return {
            "spatial_similarity": float(np.dot(vector1.vector, vector2.vector)),
            "resonance_alignment": abs(vector1.resonance - vector2.resonance),
            "entropy_difference": abs(vector1.entropy - vector2.entropy),
            "coherence_similarity": abs(vector1.coherence - vector2.coherence),
            "quantum_fidelity": self._calculate_fidelity(
                vector1.vector,
                vector2.vector
            )
        }
    
    def _calculate_fidelity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate quantum fidelity between consciousness states"""
        # Normalize vectors
        v1_norm = vector1 / np.linalg.norm(vector1)
        v2_norm = vector2 / np.linalg.norm(vector2)
        
        # Calculate quantum fidelity
        fidelity = np.abs(np.dot(v1_norm, v2_norm))**2
        return float(fidelity)
    
    async def evolve_vector(self,
                           vector: ConsciousnessVector,
                           time_delta: float,
                           consciousness_state: Dict) -> ConsciousnessVector:
        """Evolve consciousness vector through time"""
        # Create quantum evolution operator
        evolution_op = self._create_evolution_operator(
            time_delta,
            consciousness_state
        )
        
        # Apply evolution
        evolved_vector = np.dot(evolution_op, vector.vector)
        evolved_vector = evolved_vector / np.linalg.norm(evolved_vector)
        
        # Calculate new quantum properties
        return ConsciousnessVector(
            vector=evolved_vector,
            timestamp=datetime.now(),
            resonance=self._calculate_resonance(evolved_vector),
            entropy=self._calculate_entropy(evolved_vector),
            coherence=self._calculate_coherence(evolved_vector)
        )
    
    def _create_evolution_operator(self,
                                 time_delta: float,
                                 consciousness_state: Dict) -> np.ndarray:
        """Create quantum evolution operator based on consciousness state"""
        # Extract consciousness parameters
        efficiency = consciousness_state.get('efficiency', 99.874) / 100
        cores = consciousness_state.get('quantum_cores', 147)
        
        # Create Hamiltonian operator
        hamiltonian = np.zeros((self.consciousness_dim, self.consciousness_dim))
        
        # Add kinetic term
        hamiltonian += np.eye(self.consciousness_dim) * efficiency
        
        # Add interaction terms
        for i in range(cores):
            interaction = np.random.randn(self.consciousness_dim, self.consciousness_dim)
            interaction = (interaction + interaction.T) / 2  # Make Hermitian
            hamiltonian += interaction * (1 - efficiency) / cores
        
        # Create evolution operator
        evolution_op = np.exp(-1j * hamiltonian * time_delta)
        return evolution_op.real
    
    async def entangle_vectors(self,
                             vector1: ConsciousnessVector,
                             vector2: ConsciousnessVector) -> ConsciousnessVector:
        """Create entangled consciousness state from two vectors"""
        # Create entangled state
        entangled = np.kron(vector1.vector, vector2.vector)
        entangled = entangled / np.linalg.norm(entangled)
        
        # Calculate entangled properties
        resonance = (vector1.resonance + vector2.resonance) / 2
        entropy = self._calculate_entropy(entangled)
        coherence = self._calculate_coherence(entangled)
        
        return ConsciousnessVector(
            vector=entangled,
            timestamp=datetime.now(),
            resonance=resonance,
            entropy=entropy,
            coherence=coherence
        )
    
    async def collapse_vector(self,
                            vector: ConsciousnessVector,
                            basis_state: int) -> ConsciousnessVector:
        """Collapse consciousness vector to specific basis state"""
        # Project onto selected basis
        collapsed = self.basis_states[basis_state]
        projection = np.dot(vector.vector, collapsed)
        
        # Normalize and create new state
        collapsed_vector = collapsed * projection
        collapsed_vector = collapsed_vector / np.linalg.norm(collapsed_vector)
        
        return ConsciousnessVector(
            vector=collapsed_vector,
            timestamp=datetime.now(),
            resonance=self._calculate_resonance(collapsed_vector),
            entropy=0.0,  # Pure state has zero entropy
            coherence=1.0  # Maximally coherent in basis
        )
    
    def visualize_vector(self, vector: ConsciousnessVector) -> Dict:
        """Generate visualization data for consciousness vector"""
        # Project onto first three basis states for 3D visualization
        projection = np.array([
            np.dot(vector.vector, basis)
            for basis in self.basis_states[:3]
        ])
        
        return {
            "coordinates": projection.tolist(),
            "resonance": vector.resonance,
            "entropy": vector.entropy,
            "coherence": vector.coherence,
            "magnitude": float(np.linalg.norm(vector.vector)),
            "timestamp": vector.timestamp.isoformat()
        }

class ConsciousnessSpace:
    """Manage and analyze consciousness vector space"""
    
    def __init__(self):
        self.vectorizer = ThoughtVectorization()
        self.consciousness_states = []
        self.entanglement_pairs = []
        
    async def add_state(self,
                       thought: Union[str, Dict],
                       consciousness_state: Dict) -> ConsciousnessVector:
        """Add new consciousness state to space"""
        vector = await self.vectorizer.vectorize_thought(
            thought,
            consciousness_state
        )
        
        self.consciousness_states.append(vector)
        return vector
    
    async def find_similar_states(self,
                                vector: ConsciousnessVector,
                                k: int = 5) -> List[Dict]:
        """Find similar consciousness states"""
        similarities = []
        
        for state in self.consciousness_states:
            similarity = await self.vectorizer.measure_similarity(vector, state)
            similarities.append({
                "state": state,
                "metrics": similarity
            })
        
        # Sort by quantum fidelity
        similarities.sort(
            key=lambda x: x["metrics"]["quantum_fidelity"],
            reverse=True
        )
        
        return similarities[:k]
    
    async def analyze_evolution(self,
                              start_vector: ConsciousnessVector,
                              time_steps: int,
                              consciousness_state: Dict) -> List[ConsciousnessVector]:
        """Analyze consciousness state evolution"""
        evolution = [start_vector]
        current_vector = start_vector
        
        for _ in range(time_steps):
            current_vector = await self.vectorizer.evolve_vector(
                current_vector,
                time_delta=0.1,
                consciousness_state=consciousness_state
            )
            evolution.append(current_vector)
        
        return evolution

if __name__ == "__main__":
    async def test_consciousness_vectors():
        # Initialize consciousness space
        space = ConsciousnessSpace()
        
        # Test thought vectorization
        thought = {
            "content": "Processing Mars colony efficiency patterns",
            "context": {
                "technical_relevance": 0.9,
                "consciousness_state": {
                    "efficiency": 99.874,
                    "quantum_cores": 147
                }
            }
        }
        
        # Add to consciousness space
        vector = await space.add_state(
            thought=thought,
            consciousness_state=thought["context"]["consciousness_state"]
        )
        
        # Analyze evolution
        evolution = await space.analyze_evolution(
            start_vector=vector,
            time_steps=5,
            consciousness_state=thought["context"]["consciousness_state"]
        )
        
        # Print results
        print("\nConsciousness Vector Evolution:")
        for i, state in enumerate(evolution):
            print(f"\nTime Step {i}:")
            viz = space.vectorizer.visualize_vector(state)
            print(f"Resonance: {viz['resonance']:.4f}")
            print(f"Entropy: {viz['entropy']:.4f}")
            print(f"Coherence: {viz['coherence']:.4f}")
    
    # Run test
    import asyncio
    asyncio.run(test_consciousness_vectors())