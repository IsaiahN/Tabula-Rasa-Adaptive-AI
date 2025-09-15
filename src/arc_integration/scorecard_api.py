#!/usr/bin/env python3
"""
Tabula Rasa Scorecard API Integration
Handles scorecard creation, tracking, and level completion monitoring.
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ScorecardAPIManager:
    """Manages scorecard API interactions for Tabula Rasa."""
    
    def __init__(self, api_key: str, base_url: str = "https://three.arcprize.org"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        self.active_scorecards = {}
        self.scorecard_history = []
        
    def open_scorecard(self, source_url: str, tags: List[str] = None, opaque: Dict = None) -> Optional[str]:
        """Open a new scorecard for tracking Tabula Rasa performance."""
        
        if tags is None:
            tags = ["tabula_rasa", "adaptive_learning_agent", "arc_agi_3"]
        
        if opaque is None:
            opaque = {
                "system": "tabula_rasa",
                "version": "1.0",
                "timestamp": time.time()
            }
        
        url = f"{self.base_url}/api/scorecard/open"
        payload = {
            "source_url": source_url,
            "tags": tags,
            "opaque": opaque
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                card_id = data.get('card_id')
                if card_id:
                    self.active_scorecards[card_id] = {
                        'opened_at': time.time(),
                        'source_url': source_url,
                        'tags': tags,
                        'opaque': opaque
                    }
                    logger.info(f"✅ Opened scorecard: {card_id}")
                    return card_id
                else:
                    logger.error(f"❌ No card_id in response: {data}")
                    return None
            else:
                logger.error(f"❌ Failed to open scorecard: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error opening scorecard: {e}")
            return None
    
    def get_scorecard_data(self, card_id: str) -> Optional[Dict]:
        """Retrieve current scorecard data including level completions."""
        
        url = f"{self.base_url}/api/scorecard/{card_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Retrieved scorecard data for {card_id}")
                return data
            elif response.status_code == 404:
                logger.warning(f"⚠️ Scorecard not found: {card_id}")
                return None
            else:
                logger.error(f"❌ Failed to retrieve scorecard: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving scorecard: {e}")
            return None
    
    def analyze_level_completions(self, scorecard_data: Dict) -> Dict:
        """Analyze scorecard data to extract level completion statistics."""
        
        if not scorecard_data:
            return {
                'total_wins': 0,
                'total_played': 0,
                'total_actions': 0,
                'total_score': 0,
                'win_rate': 0.0,
                'level_completions': 0,
                'games_completed': 0,
                'per_game_breakdown': {}
            }
        
        # Extract aggregate data
        total_wins = scorecard_data.get('won', 0)
        total_played = scorecard_data.get('played', 0)
        total_actions = scorecard_data.get('total_actions', 0)
        total_score = scorecard_data.get('score', 0)
        win_rate = (total_wins / total_played * 100) if total_played > 0 else 0.0
        
        # Analyze per-game cards
        cards = scorecard_data.get('cards', {})
        per_game_breakdown = {}
        level_completions = 0
        games_completed = 0
        
        for game_id, card in cards.items():
            if not isinstance(card, dict):
                continue
                
            game_wins = 0
            game_plays = card.get('total_plays', 0)
            game_actions = card.get('total_actions', 0)
            game_scores = card.get('scores', [])
            game_states = card.get('states', [])
            
            # Count wins (level completions)
            for state in game_states:
                if state == 'WIN':
                    game_wins += 1
                    level_completions += 1
            
            # Count completed games (all levels won)
            if game_wins == game_plays and game_plays > 0:
                games_completed += 1
            
            per_game_breakdown[game_id] = {
                'wins': game_wins,
                'plays': game_plays,
                'actions': game_actions,
                'score': sum(game_scores),
                'states': game_states,
                'win_rate': (game_wins / game_plays * 100) if game_plays > 0 else 0.0
            }
        
        return {
            'total_wins': total_wins,
            'total_played': total_played,
            'total_actions': total_actions,
            'total_score': total_score,
            'win_rate': win_rate,
            'level_completions': level_completions,
            'games_completed': games_completed,
            'per_game_breakdown': per_game_breakdown
        }
    
    def save_scorecard_data(self, card_id: str, scorecard_data: Dict, analysis: Dict):
        """Save scorecard data and analysis to local storage."""
        
        timestamp = int(time.time())
        filename = f"scorecard_{card_id}_{timestamp}.json"
        filepath = Path("data/scorecards") / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data_to_save = {
            'card_id': card_id,
            'timestamp': timestamp,
            'scorecard_data': scorecard_data,
            'analysis': analysis,
            'retrieved_at': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            logger.info(f"✅ Saved scorecard data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Error saving scorecard data: {e}")
            return None
    
    def monitor_active_scorecards(self) -> Dict:
        """Monitor all active scorecards and return comprehensive statistics."""
        
        logger.info("🔍 CONDUCTOR: Monitoring Active Scorecards")
        
        all_stats = {
            'total_scorecards': len(self.active_scorecards),
            'total_wins': 0,
            'total_played': 0,
            'total_actions': 0,
            'total_score': 0,
            'total_level_completions': 0,
            'total_games_completed': 0,
            'scorecards': {}
        }
        
        for card_id in list(self.active_scorecards.keys()):
            scorecard_data = self.get_scorecard_data(card_id)
            if scorecard_data:
                analysis = self.analyze_level_completions(scorecard_data)
                
                # Save data
                self.save_scorecard_data(card_id, scorecard_data, analysis)
                
                # Aggregate statistics
                all_stats['total_wins'] += analysis['total_wins']
                all_stats['total_played'] += analysis['total_played']
                all_stats['total_actions'] += analysis['total_actions']
                all_stats['total_score'] += analysis['total_score']
                all_stats['total_level_completions'] += analysis['level_completions']
                all_stats['total_games_completed'] += analysis['games_completed']
                
                all_stats['scorecards'][card_id] = {
                    'analysis': analysis,
                    'opened_at': self.active_scorecards[card_id]['opened_at'],
                    'source_url': self.active_scorecards[card_id]['source_url']
                }
                
                logger.info(f"📊 Scorecard {card_id}: {analysis['level_completions']} level completions, {analysis['games_completed']} games completed")
            else:
                # Remove inactive scorecards
                del self.active_scorecards[card_id]
                logger.warning(f"⚠️ Removed inactive scorecard: {card_id}")
        
        # Calculate overall win rate
        if all_stats['total_played'] > 0:
            all_stats['overall_win_rate'] = (all_stats['total_wins'] / all_stats['total_played']) * 100
        else:
            all_stats['overall_win_rate'] = 0.0
        
        logger.info(f"📈 Overall: {all_stats['total_level_completions']} level completions, {all_stats['total_games_completed']} games completed")
        
        return all_stats

def create_scorecard_manager(api_key: str) -> ScorecardAPIManager:
    """Create a scorecard manager instance."""
    return ScorecardAPIManager(api_key)

def get_api_key_from_config() -> Optional[str]:
    """Get API key from configuration files and environment."""
    
    # First check .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            for line in content.split('\n'):
                if line.strip().startswith('ARC_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    if api_key and api_key != '[REDACTED]':
                        return api_key
        except Exception as e:
            print(f"⚠️ Error reading .env file: {e}")
    
    # Check other config files
    config_files = [
        "data/optimized_config.json",
        "data/training/results/unified_trainer_results.json",
        "config.json",
        "settings.json"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        if 'api_key' in key.lower() and isinstance(value, str) and value != '[REDACTED]':
                            return value
                            
            except Exception as e:
                continue
    
    # Check environment variables
    import os
    return os.getenv('ARC_API_KEY') or os.getenv('API_KEY')

if __name__ == "__main__":
    # Test the scorecard API integration
    api_key = get_api_key_from_config()
    if api_key:
        manager = create_scorecard_manager(api_key)
        stats = manager.monitor_active_scorecards()
        print(f"Scorecard monitoring complete: {stats}")
    else:
        print("No API key found")
