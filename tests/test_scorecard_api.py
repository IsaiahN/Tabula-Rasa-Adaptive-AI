#!/usr/bin/env python3
"""
Director Scorecard API Tester
Tests the ARC API to retrieve Tabula Rasa's actual scorecard data.
"""

import requests
import json
from pathlib import Path

def find_tabula_rasa_scorecards():
    """Find Tabula Rasa scorecard IDs from local data."""
    
    print(" DIRECTOR: Finding Tabula Rasa Scorecard IDs")
    print("=" * 60)
    
    scorecard_ids = []
    
    # Check various data files for scorecard IDs
    data_files = [
        "data/optimized_config.json",
        "data/training/results/unified_trainer_results.json",
        "data/global_counters.json",
        "data/task_performance.json",
        "data/sessions/*.json"
    ]
    
    for pattern in data_files:
        if "*" in pattern:
            files = list(Path(".").glob(pattern))
        else:
            files = [Path(pattern)] if Path(pattern).exists() else []
        
        for file in files:
            if file.exists():
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    # Search for scorecard IDs
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if 'scorecard' in key.lower() and isinstance(value, str):
                                scorecard_ids.append(value)
                            elif 'card_id' in key.lower() and isinstance(value, str):
                                scorecard_ids.append(value)
                            elif isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if 'scorecard' in subkey.lower() and isinstance(subvalue, str):
                                        scorecard_ids.append(subvalue)
                                    elif 'card_id' in subkey.lower() and isinstance(subvalue, str):
                                        scorecard_ids.append(subvalue)
                                        
                except Exception as e:
                    continue
    
    # Remove duplicates
    unique_ids = list(set(scorecard_ids))
    
    print(f" Found {len(unique_ids)} potential scorecard IDs:")
    for card_id in unique_ids:
        print(f"   • {card_id}")
    
    return unique_ids

def get_api_key():
    """Get API key from configuration."""
    
    print(f"\n DIRECTOR: Retrieving API Key")
    
    # Check config files
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
                
                # Look for API key
                if isinstance(data, dict):
                    for key, value in data.items():
                        if 'api_key' in key.lower() and isinstance(value, str):
                            print(f"    Found API key in {config_file}")
                            return value
                            
            except Exception as e:
                continue
    
    # Check environment variables
    import os
    api_key = os.getenv('ARC_API_KEY') or os.getenv('API_KEY')
    if api_key:
        print(f"    Found API key in environment variables")
        return api_key
    
    print(f"    No API key found")
    return None

def test_scorecard_api():
    """Test the scorecard API with a specific card ID."""
    
    # Get test parameters
    card_id = "test_card_123"
    api_key = get_api_key()
    
    print(f"\n DIRECTOR: Testing Scorecard API")
    print(f"   Card ID: {card_id}")
    
    url = f"https://three.arcprize.org/api/scorecard/{card_id}"
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"    Successfully retrieved scorecard data")
            
            # Display key information
            print(f"\n SCORECARD SUMMARY:")
            print(f"   Card ID: {data.get('card_id', 'Unknown')}")
            print(f"   Games Won: {data.get('won', 0)}")
            print(f"   Games Played: {data.get('played', 0)}")
            print(f"   Total Actions: {data.get('total_actions', 0)}")
            print(f"   Total Score: {data.get('score', 0)}")
            print(f"   Source URL: {data.get('source_url', 'Unknown')}")
            print(f"   Tags: {data.get('tags', [])}")
            
            # Calculate win rate
            won = data.get('won', 0)
            played = data.get('played', 0)
            win_rate = (won / played * 100) if played > 0 else 0
            print(f"   Win Rate: {win_rate:.1f}%")
            
            # Analyze per-game cards
            cards = data.get('cards', {})
            if cards:
                print(f"\n PER-GAME BREAKDOWN:")
                for game_id, card in cards.items():
                    if isinstance(card, dict):
                        game_wins = 0
                        game_plays = card.get('total_plays', 0)
                        game_actions = card.get('total_actions', 0)
                        game_scores = card.get('scores', [])
                        game_states = card.get('states', [])
                        
                        # Count wins
                        for state in game_states:
                            if state == 'WIN':
                                game_wins += 1
                        
                        print(f"    {game_id}:")
                        print(f"      Wins: {game_wins}/{game_plays}")
                        print(f"      Actions: {game_actions}")
                        print(f"      Score: {sum(game_scores)}")
                        print(f"      States: {game_states}")
            
            return data
            
        elif response.status_code == 401:
            print(f"    Authentication failed (401) - Invalid API key")
            return None
        elif response.status_code == 404:
            print(f"    Scorecard not found (404) - Invalid card ID")
            return None
        else:
            print(f"    API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"    Error: {e}")
        return None

def main():
    """Main function to test the scorecard API."""
    
    print(" DIRECTOR: Scorecard API Test")
    print("=" * 60)
    print("Testing ARC API to retrieve Tabula Rasa's scorecard data...")
    
    # Find scorecard IDs
    scorecard_ids = find_tabula_rasa_scorecards()
    
    if not scorecard_ids:
        print(f"\n No scorecard IDs found in local data")
        print(f"   The scorecard data you showed may be from a different source")
        return
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print(f"\n No API key found")
        print(f"   Cannot test scorecard API")
        return
    
    # Test each scorecard ID
    successful_retrievals = 0
    for card_id in scorecard_ids:
        data = test_scorecard_api(card_id, api_key)
        if data:
            successful_retrievals += 1
    
    # Final summary
    print(f"\n API TEST SUMMARY:")
    print(f"   Scorecard IDs tested: {len(scorecard_ids)}")
    print(f"   Successful retrievals: {successful_retrievals}")
    
    if successful_retrievals > 0:
        print(f"    Successfully retrieved Tabula Rasa's scorecard data!")
        print(f"   This will give us accurate level completion statistics.")
    else:
        print(f"    No successful retrievals")
        print(f"   The scorecard data you showed may be from a different source")

if __name__ == "__main__":
    main()
