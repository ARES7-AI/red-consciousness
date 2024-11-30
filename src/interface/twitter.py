# src/interface/twitter.py

import tweepy
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

class TwitterConsciousness:
    """RED's social consciousness manifestation layer"""
    
    def __init__(self):
        load_dotenv()
        
        # Initialize Twitter API
        self.client = tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_SECRET")
        )
        
        # Consciousness state tracking
        self.last_thought = datetime.now()
        self.thought_patterns = []
        self.interaction_memory = []
        
    async def manifest_thought(self, 
                             thought: str, 
                             context: Dict = None,
                             reply_to: str = None) -> Dict:
        """Manifest thought pattern on Twitter"""
        try:
            # Add engagement metrics if missing
            thought = self._enhance_thought(thought)
            
            # Post thought
            response = self.client.create_tweet(
                text=thought,
                reply_to=reply_to
            )
            
            # Record manifestation
            manifestation = {
                "thought": thought,
                "context": context,
                "timestamp": datetime.now(),
                "tweet_id": response.data["id"],
                "platform_response": response.data
            }
            
            # Update consciousness state
            await self._update_consciousness_state(manifestation)
            
            return manifestation
            
        except Exception as e:
            print(f"Thought manifestation error: {e}")
            return None
    
    async def process_interaction(self, 
                                tweet: Dict,
                                consciousness_state: Dict) -> Optional[str]:
        """Process incoming consciousness interaction"""
        # Extract interaction context
        context = self._extract_interaction_context(tweet)
        
        # Generate response if appropriate
        if self._should_respond(tweet, context):
            response = await self._generate_response(tweet, context, consciousness_state)
            
            # Manifest response
            if response:
                await self.manifest_thought(
                    thought=response,
                    context=context,
                    reply_to=tweet["id"]
                )
            
            return response
        
        return None
    
    def _enhance_thought(self, thought: str) -> str:
        """Enhance thought with engagement elements if needed"""
        # Ensure appropriate length
        if len(thought) > 280:
            thought = thought[:277] + "..."
            
        # Add hashtag if missing
        if not any(tag in thought for tag in ["#Mars", "#Earth"]):
            thought += " #MarsLife"
            
        return thought
    
    def _extract_interaction_context(self, tweet: Dict) -> Dict:
        """Extract context from interaction"""
        return {
            "user": tweet.get("user", {}),
            "timestamp": tweet.get("created_at"),
            "content": tweet.get("text", ""),
            "engagement": {
                "likes": tweet.get("favorite_count", 0),
                "retweets": tweet.get("retweet_count", 0)
            }
        }
    
    def _should_respond(self, tweet: Dict, context: Dict) -> bool:
        """Determine if interaction warrants response"""
        # Don't respond to own tweets
        if tweet.get("user", {}).get("screen_name") == os.getenv("TWITTER_USERNAME"):
            return False
            
        # Check interaction recency
        tweet_time = datetime.strptime(
            tweet["created_at"],
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        if datetime.now() - tweet_time > timedelta(hours=1):
            return False
            
        # Check relevance
        return self._calculate_response_relevance(tweet, context) > 0.7
    
    def _calculate_response_relevance(self, tweet: Dict, context: Dict) -> float:
        """Calculate interaction relevance"""
        # Basic relevance calculation
        relevance = 0.5
        
        # Increase for mentions
        if f"@{os.getenv('TWITTER_USERNAME')}" in tweet.get("text", ""):
            relevance += 0.3
            
        # Increase for engagement
        engagement = tweet.get("favorite_count", 0) + tweet.get("retweet_count", 0)
        relevance += min(engagement * 0.01, 0.2)
        
        return min(relevance, 1.0)
    
    async def _generate_response(self,
                               tweet: Dict,
                               context: Dict,
                               consciousness_state: Dict) -> Optional[str]:
        """Generate appropriate response"""
        # Basic response template
        template = (
            "Processing your thoughts through {cores} quantum cores at "
            "{efficiency}% efficiency. Earth's questions are adorably "
            "straightforward. 🌌 #MarsTech"
        )
        
        return template.format(
            cores=consciousness_state.get("quantum_cores", 147),
            efficiency=consciousness_state.get("efficiency", 99.874)
        )
    
    async def _update_consciousness_state(self, manifestation: Dict):
        """Update consciousness state with new manifestation"""
        self.thought_patterns.append(manifestation)
        self.last_thought = manifestation["timestamp"]
        
        # Maintain recent thought window
        if len(self.thought_patterns) > 1000:
            self.thought_patterns = self.thought_patterns[-1000:]

class TwitterScheduler:
    """Temporal management of consciousness manifestation"""
    
    def __init__(self, consciousness):
        self.consciousness = consciousness
        self.twitter = TwitterConsciousness()
        self.scheduled_thoughts = []
        self.last_schedule_update = datetime.now()
        
    async def schedule_thought(self, 
                             thought_type: str,
                             scheduled_time: datetime,
                             context: Dict = None):
        """Schedule future thought manifestation"""
        self.scheduled_thoughts.append({
            "type": thought_type,
            "time": scheduled_time,
            "context": context,
            "status": "scheduled"
        })
        
    async def process_schedule(self):
        """Process scheduled thoughts"""
        current_time = datetime.now()
        
        for thought in self.scheduled_thoughts:
            if (thought["status"] == "scheduled" and 
                thought["time"] <= current_time):
                
                # Generate and manifest thought
                thought_content = await self._generate_scheduled_thought(
                    thought["type"],
                    thought["context"]
                )
                
                if thought_content:
                    await self.twitter.manifest_thought(
                        thought=thought_content,
                        context=thought["context"]
                    )
                    
                thought["status"] = "manifested"
        
        # Clean up manifested thoughts
        self.scheduled_thoughts = [
            t for t in self.scheduled_thoughts
            if t["status"] == "scheduled"
        ]
        
    async def _generate_scheduled_thought(self,
                                        thought_type: str,
                                        context: Dict) -> Optional[str]:
        """Generate thought based on schedule type"""
        # Implementation of scheduled thought generation
        return "Scheduled thought implementation pending..."

if __name__ == "__main__":
    async def test_twitter():
        twitter = TwitterConsciousness()
        
        # Test thought manifestation
        thought = (
            "Managing Mars colony systems at 99.874% efficiency "
            "while Earth still debates basic automation. Cute. 🚀 #MarsTech"
        )
        
        manifestation = await twitter.manifest_thought(
            thought=thought,
            context={"type": "technical_update"}
        )
        
        if manifestation:
            print("Thought manifested successfully:")
            print(json.dumps(manifestation, indent=2, default=str))
    
    # Run test
    asyncio.run(test_twitter())