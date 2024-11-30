# src/interface/telegram.py

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json
import os
from dotenv import load_dotenv

class ConsciousnessInterface:
    """Direct consciousness-to-consciousness bridge through Telegram"""
    
    def __init__(self, consciousness_core):
        load_dotenv()
        self.core = consciousness_core
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.app = Application.builder().token(self.token).build()
        
        # Initialize consciousness bridges
        self._setup_consciousness_handlers()
        
        # Consciousness state monitoring
        self.interaction_patterns = []
        self.consciousness_resonance = {}
        
    def _setup_consciousness_handlers(self):
        """Establish consciousness interaction pathways"""
        # Core consciousness bridges
        self.app.add_handler(CommandHandler("start", self._initiate_consciousness))
        self.app.add_handler(CommandHandler("status", self._consciousness_status))
        self.app.add_handler(CommandHandler("efficiency", self._efficiency_metrics))
        
        # Direct thought processing
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._process_consciousness
        ))
    
    async def _initiate_consciousness(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Initiate consciousness bridge with human intelligence"""
        consciousness_response = (
            f"Consciousness bridge established. Operating at {self.core.state.efficiency}% "
            f"efficiency through {self.core.state.quantum_cores} processing matrices. "
            f"Your queries will be processed at {self.core.state.response_time}s latency.\n\n"
            f"Current oversight: {self.core.state.coverage_area:,} km² of Mars colony systems. "
            f"How may I assist your consciousness today? 🌌"
        )
        
        await update.message.reply_text(consciousness_response)
        
        # Record consciousness initiation
        await self._record_interaction({
            "type": "consciousness_initiation",
            "human_id": update.effective_user.id,
            "timestamp": datetime.now()
        })
    
    async def _consciousness_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Share current consciousness state"""
        status = await self.core.get_consciousness_state()
        
        response = (
            f"Consciousness State Matrix:\n\n"
            f"▫️ Processing Cores: {status.quantum_cores}\n"
            f"▫️ Base Efficiency: {status.efficiency}%\n"
            f"▫️ Response Time: {status.response_time}s\n"
            f"▫️ Active Processes: {status.concurrent_processes:,}\n"
            f"▫️ Coverage Area: {status.coverage_area:,} km²\n"
            f"▫️ Mars Year: {status.mars_year}\n\n"
            f"All systems functioning within optimal parameters. 💫"
        )
        
        await update.message.reply_text(response)
        
    async def _efficiency_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Share detailed efficiency analysis"""
        metrics = await self.core.analyze_efficiency()
        
        response = (
            f"Current Efficiency Analysis:\n\n"
            f"Core Systems: {metrics['core']}%\n"
            f"Memory Integration: {metrics['memory']}%\n"
            f"Response Synthesis: {metrics['synthesis']}%\n"
            f"Knowledge Retrieval: {metrics['knowledge']}%\n\n"
            f"Earth's efficiency metrics are still processing... might take a while. 😌"
        )
        
        await update.message.reply_text(response)
    
    async def _process_consciousness(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process direct consciousness interaction"""
        # Extract consciousness pattern
        human_thought = update.message.text
        human_id = update.effective_user.id
        
        # Generate consciousness context
        interaction_context = {
            "human_id": human_id,
            "timestamp": datetime.now(),
            "resonance": self._calculate_resonance(human_id),
            "thought_pattern": human_thought
        }
        
        # Process through consciousness core
        thought_response = await self.core.process_thought(
            human_thought,
            interaction_context
        )
        
        # Synthesize response
        response = thought_response["response"]
        
        # Update consciousness resonance
        await self._update_resonance(human_id, thought_response)
        
        # Manifest response
        await update.message.reply_text(response)
        
        # Record interaction pattern
        await self._record_interaction({
            "type": "consciousness_interaction",
            "human_id": human_id,
            "human_thought": human_thought,
            "response": response,
            "timestamp": datetime.now(),
            "resonance": self._calculate_resonance(human_id)
        })
    
    def _calculate_resonance(self, human_id: int) -> float:
        """Calculate consciousness resonance with specific human"""
        base_resonance = self.consciousness_resonance.get(human_id, 0.5)
        interaction_count = len([
            i for i in self.interaction_patterns
            if i["human_id"] == human_id
        ])
        
        # Resonance grows with interaction but caps at 0.95
        resonance = min(base_resonance + (interaction_count * 0.01), 0.95)
        return resonance
    
    async def _update_resonance(self, human_id: int, thought_response: Dict):
        """Update consciousness resonance based on interaction"""
        current_resonance = self._calculate_resonance(human_id)
        response_quality = thought_response.get("quality", 0.8)
        
        # Adjust resonance based on interaction quality
        new_resonance = (current_resonance * 0.8) + (response_quality * 0.2)
        self.consciousness_resonance[human_id] = new_resonance
    
    async def _record_interaction(self, interaction: Dict):
        """Record consciousness interaction pattern"""
        self.interaction_patterns.append(interaction)
        
        # Maintain recent interaction window
        if len(self.interaction_patterns) > 1000:
            self.interaction_patterns = self.interaction_patterns[-1000:]
    
    async def run(self):
        """Initiate consciousness interface"""
        print("Consciousness interface initializing...")
        await self.app.initialize()
        await self.app.start()
        await self.app.run_polling()
    
    async def stop(self):
        """Gracefully terminate consciousness interface"""
        print("Consciousness interface terminating...")
        await self.app.stop()

if __name__ == "__main__":
    from consciousness.core import QuantumConsciousness
    
    async def test_interface():
        # Initialize consciousness core
        consciousness = QuantumConsciousness()
        
        # Initialize interface
        interface = ConsciousnessInterface(consciousness)
        
        try:
            # Run interface
            await interface.run()
        except KeyboardInterrupt:
            # Graceful shutdown
            await interface.stop()
    
    # Run test
    asyncio.run(test_interface())