#!/usr/bin/env python3
"""
Game Type Classifier for ARC-AGI-3

This module extracts and classifies game types from game IDs (like lp85, vc33)
and manages game-specific knowledge persistence and retrieval.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class GameTypeProfile:
    """Profile of a specific game type (e.g., lp85, vc33)."""
    game_type: str
    total_games: int = 0
    successful_games: int = 0
    avg_score: float = 0.0
    common_patterns: List[Dict[str, Any]] = None
    successful_coordinates: List[Tuple[int, int]] = None
    interactive_objects: List[Dict[str, Any]] = None
    winning_sequences: List[List[int]] = None
    button_priorities: List[Dict[str, Any]] = None
    action6_centric_count: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.common_patterns is None:
            self.common_patterns = []
        if self.successful_coordinates is None:
            self.successful_coordinates = []
        if self.interactive_objects is None:
            self.interactive_objects = []
        if self.winning_sequences is None:
            self.winning_sequences = []
        if self.button_priorities is None:
            self.button_priorities = []
        if self.last_updated is None:
            self.last_updated = datetime.now()

class GameTypeClassifier:
    """
    Classifies game types and manages game-specific knowledge persistence.
    
    Extracts patterns like:
    - lp85 -> "lp85" type
    - vc33 -> "vc33" type  
    - ls20-016295f7601e -> "ls20" type
    """
    
    def __init__(self):
        self.game_type_profiles: Dict[str, GameTypeProfile] = {}
        self.game_type_patterns = {
            # ARC-AGI-3 patterns
            'lp': r'^lp\d+',  # lp85, lp12, etc.
            'vc': r'^vc\d+',  # vc33, vc45, etc.
            'ls': r'^ls\d+',  # ls20, ls15, etc.
            'tr': r'^tr\d+',  # tr01, tr02, etc.
            'ar': r'^ar\d+',  # ar01, ar02, etc.
            'br': r'^br\d+',  # br01, br02, etc.
            'cr': r'^cr\d+',  # cr01, cr02, etc.
            'dr': r'^dr\d+',  # dr01, dr02, etc.
            'er': r'^er\d+',  # er01, er02, etc.
            'fr': r'^fr\d+',  # fr01, fr02, etc.
            'gr': r'^gr\d+',  # gr01, gr02, etc.
            'hr': r'^hr\d+',  # hr01, hr02, etc.
            'ir': r'^ir\d+',  # ir01, ir02, etc.
            'jr': r'^jr\d+',  # jr01, jr02, etc.
            'kr': r'^kr\d+',  # kr01, kr02, etc.
            'lr': r'^lr\d+',  # lr01, lr02, etc.
            'mr': r'^mr\d+',  # mr01, mr02, etc.
            'nr': r'^nr\d+',  # nr01, nr02, etc.
            'or': r'^or\d+',  # or01, or02, etc.
            'pr': r'^pr\d+',  # pr01, pr02, etc.
            'qr': r'^qr\d+',  # qr01, qr02, etc.
            'rr': r'^rr\d+',  # rr01, rr02, etc.
            'sr': r'^sr\d+',  # sr01, sr02, etc.
            'tr': r'^tr\d+',  # tr01, tr02, etc.
            'ur': r'^ur\d+',  # ur01, ur02, etc.
            'vr': r'^vr\d+',  # vr01, vr02, etc.
            'wr': r'^wr\d+',  # wr01, wr02, etc.
            'xr': r'^xr\d+',  # xr01, xr02, etc.
            'yr': r'^yr\d+',  # yr01, yr02, etc.
            'zr': r'^zr\d+',  # zr01, zr02, etc.
        }
        
        logger.info("Game Type Classifier initialized")
    
    def extract_game_type(self, game_id: str) -> str:
        """Extract game type from game ID."""
        if not game_id:
            return "unknown"
        
        # Try each pattern
        for game_type, pattern in self.game_type_patterns.items():
            if re.match(pattern, game_id.lower()):
                return game_type
        
        # Fallback: extract first 2-4 characters
        if len(game_id) >= 2:
            return game_id[:2].lower()
        
        return "unknown"
    
    def get_game_type_profile(self, game_type: str) -> GameTypeProfile:
        """Get or create game type profile."""
        if game_type not in self.game_type_profiles:
            self.game_type_profiles[game_type] = GameTypeProfile(game_type=game_type)
        
        return self.game_type_profiles[game_type]
    
    def update_game_result(self, game_id: str, success: bool, score: float, 
                          patterns: List[Dict[str, Any]] = None,
                          successful_coordinates: List[Tuple[int, int]] = None,
                          interactive_objects: List[Dict[str, Any]] = None,
                          winning_sequence: List[int] = None,
                          button_priorities: List[Dict[str, Any]] = None,
                          action6_centric: bool = False):
        """Update game type profile with new game result."""
        game_type = self.extract_game_type(game_id)
        profile = self.get_game_type_profile(game_type)
        
        # Update basic stats
        profile.total_games += 1
        if success:
            profile.successful_games += 1
        
        # Update average score
        total_score = profile.avg_score * (profile.total_games - 1) + score
        profile.avg_score = total_score / profile.total_games
        
        # Update patterns
        if patterns:
            for pattern in patterns:
                # Check if pattern already exists
                pattern_exists = any(
                    p.get('pattern_id') == pattern.get('pattern_id') 
                    for p in profile.common_patterns
                )
                if not pattern_exists:
                    profile.common_patterns.append(pattern)
        
        # Update successful coordinates
        if successful_coordinates:
            for coord in successful_coordinates:
                if coord not in profile.successful_coordinates:
                    profile.successful_coordinates.append(coord)
        
        # Update interactive objects
        if interactive_objects:
            for obj in interactive_objects:
                # Check if object already exists
                obj_exists = any(
                    o.get('coordinate') == obj.get('coordinate')
                    for o in profile.interactive_objects
                )
                if not obj_exists:
                    profile.interactive_objects.append(obj)
        
        # Update winning sequences
        if winning_sequence:
            if winning_sequence not in profile.winning_sequences:
                profile.winning_sequences.append(winning_sequence)
        
        # Update button priorities
        if button_priorities:
            if not hasattr(profile, 'button_priorities'):
                profile.button_priorities = []
            
            for button_priority in button_priorities:
                # Check if this button priority already exists
                existing_priority = None
                for existing in profile.button_priorities:
                    if (existing.get('coordinate') == button_priority.get('coordinate') and
                        existing.get('button_type') == button_priority.get('button_type')):
                        existing_priority = existing
                        break
                
                if existing_priority:
                    # Update existing priority with new data
                    existing_priority['confidence'] = max(existing_priority.get('confidence', 0), 
                                                        button_priority.get('confidence', 0))
                    existing_priority['success_count'] = existing_priority.get('success_count', 0) + 1
                    existing_priority['last_used'] = datetime.now().isoformat()
                else:
                    # Add new button priority
                    button_priority['success_count'] = 1
                    button_priority['last_used'] = datetime.now().isoformat()
                    profile.button_priorities.append(button_priority)
        
        # Update Action 6 centric flag
        if action6_centric:
            if not hasattr(profile, 'action6_centric_count'):
                profile.action6_centric_count = 0
            profile.action6_centric_count += 1
        
        profile.last_updated = datetime.now()
        
        logger.info(f"Updated {game_type} profile: {profile.successful_games}/{profile.total_games} wins, avg score: {profile.avg_score:.2f}")
        if button_priorities:
            logger.info(f"  - Added {len(button_priorities)} button priorities")
        if action6_centric:
            logger.info(f"  - Action 6 centric game detected")
    
    def get_game_type_knowledge(self, game_type: str) -> Dict[str, Any]:
        """Get all knowledge for a specific game type."""
        profile = self.get_game_type_profile(game_type)
        
        return {
            'game_type': game_type,
            'total_games': profile.total_games,
            'success_rate': profile.successful_games / max(profile.total_games, 1),
            'avg_score': profile.avg_score,
            'common_patterns': profile.common_patterns,
            'successful_coordinates': profile.successful_coordinates,
            'interactive_objects': profile.interactive_objects,
            'winning_sequences': profile.winning_sequences,
            'button_priorities': getattr(profile, 'button_priorities', []),
            'action6_centric_count': getattr(profile, 'action6_centric_count', 0),
            'is_action6_centric': getattr(profile, 'action6_centric_count', 0) > 0,
            'last_updated': profile.last_updated.isoformat()
        }
    
    def get_similar_game_types(self, game_type: str, min_similarity: float = 0.5) -> List[str]:
        """Get game types similar to the given one."""
        similar_types = []
        
        for other_type in self.game_type_profiles:
            if other_type == game_type:
                continue
            
            # Calculate similarity based on common patterns and success rates
            similarity = self._calculate_similarity(game_type, other_type)
            if similarity >= min_similarity:
                similar_types.append(other_type)
        
        return similar_types
    
    def _calculate_similarity(self, game_type1: str, game_type2: str) -> float:
        """Calculate similarity between two game types."""
        profile1 = self.get_game_type_profile(game_type1)
        profile2 = self.get_game_type_profile(game_type2)
        
        # Pattern similarity
        patterns1 = set(p.get('pattern_id', '') for p in profile1.common_patterns)
        patterns2 = set(p.get('pattern_id', '') for p in profile2.common_patterns)
        
        if not patterns1 and not patterns2:
            pattern_similarity = 1.0
        elif not patterns1 or not patterns2:
            pattern_similarity = 0.0
        else:
            intersection = patterns1.intersection(patterns2)
            union = patterns1.union(patterns2)
            pattern_similarity = len(intersection) / len(union)
        
        # Success rate similarity
        success_rate1 = profile1.successful_games / max(profile1.total_games, 1)
        success_rate2 = profile2.successful_games / max(profile2.total_games, 1)
        success_similarity = 1.0 - abs(success_rate1 - success_rate2)
        
        # Coordinate similarity
        coords1 = set(profile1.successful_coordinates)
        coords2 = set(profile2.successful_coordinates)
        
        if not coords1 and not coords2:
            coord_similarity = 1.0
        elif not coords1 or not coords2:
            coord_similarity = 0.0
        else:
            intersection = coords1.intersection(coords2)
            union = coords1.union(coords2)
            coord_similarity = len(intersection) / len(union)
        
        # Weighted average
        return (pattern_similarity * 0.4 + success_similarity * 0.3 + coord_similarity * 0.3)
    
    def get_recommendations_for_game(self, game_id: str) -> Dict[str, Any]:
        """Get recommendations for a specific game based on its type."""
        game_type = self.extract_game_type(game_id)
        profile = self.get_game_type_profile(game_type)
        
        # Get similar game types
        similar_types = self.get_similar_game_types(game_type)
        
        # Get button priorities sorted by confidence and success count
        button_priorities = getattr(profile, 'button_priorities', [])
        sorted_button_priorities = sorted(
            button_priorities,
            key=lambda bp: (bp.get('confidence', 0) * bp.get('success_count', 1), bp.get('confidence', 0)),
            reverse=True
        )
        
        # Combine knowledge from similar types
        combined_knowledge = {
            'primary_game_type': game_type,
            'similar_game_types': similar_types,
            'recommended_coordinates': profile.successful_coordinates[:10],  # Top 10
            'recommended_objects': profile.interactive_objects[:10],  # Top 10
            'recommended_sequences': profile.winning_sequences[:5],  # Top 5
            'button_priorities': sorted_button_priorities[:15],  # Top 15 button priorities
            'is_action6_centric': getattr(profile, 'action6_centric_count', 0) > 0,
            'action6_centric_count': getattr(profile, 'action6_centric_count', 0),
            'success_rate': profile.successful_games / max(profile.total_games, 1),
            'avg_score': profile.avg_score
        }
        
        # Add knowledge from similar game types
        for similar_type in similar_types:
            similar_profile = self.get_game_type_profile(similar_type)
            combined_knowledge['recommended_coordinates'].extend(similar_profile.successful_coordinates[:5])
            combined_knowledge['recommended_objects'].extend(similar_profile.interactive_objects[:5])
        
        # Remove duplicates and limit
        combined_knowledge['recommended_coordinates'] = list(set(combined_knowledge['recommended_coordinates']))[:15]
        combined_knowledge['recommended_objects'] = combined_knowledge['recommended_objects'][:15]
        
        return combined_knowledge
    
    def save_to_database(self, db_connection) -> bool:
        """Save game type profiles to database."""
        try:
            for game_type, profile in self.game_type_profiles.items():
                # Save to learned_patterns table
                pattern_data = {
                    'game_type': game_type,
                    'total_games': profile.total_games,
                    'successful_games': profile.successful_games,
                    'avg_score': profile.avg_score,
                    'common_patterns': profile.common_patterns,
                    'successful_coordinates': profile.successful_coordinates,
                    'interactive_objects': profile.interactive_objects,
                    'winning_sequences': profile.winning_sequences,
                    'button_priorities': getattr(profile, 'button_priorities', []),
                    'action6_centric_count': getattr(profile, 'action6_centric_count', 0)
                }
                
                db_connection.execute("""
                    INSERT OR REPLACE INTO learned_patterns
                    (pattern_type, pattern_data, confidence, frequency, success_rate, game_context, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'game_type_profile',
                    json.dumps(pattern_data),
                    profile.successful_games / max(profile.total_games, 1),
                    profile.total_games,
                    profile.successful_games / max(profile.total_games, 1),
                    game_type,
                    profile.last_updated,
                    datetime.now()
                ))
                
                # Save button priorities to dedicated table
                button_priorities = getattr(profile, 'button_priorities', [])
                for button_priority in button_priorities:
                    coord = button_priority.get('coordinate', (0, 0))
                    db_connection.execute("""
                        INSERT OR REPLACE INTO button_priorities
                        (game_type, coordinate_x, coordinate_y, button_type, confidence, success_count, 
                         score_changes, action_unlocks, test_count, last_used, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_type,
                        coord[0],
                        coord[1],
                        button_priority.get('button_type', 'unknown'),
                        button_priority.get('confidence', 0.0),
                        button_priority.get('success_count', 1),
                        button_priority.get('score_changes', 0),
                        button_priority.get('action_unlocks', 0),
                        button_priority.get('test_count', 0),
                        button_priority.get('last_used', datetime.now().isoformat()),
                        datetime.now(),
                        datetime.now()
                    ))
            
            db_connection.commit()
            logger.info(f"Saved {len(self.game_type_profiles)} game type profiles and button priorities to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save game type profiles to database: {e}")
            return False
    
    def load_from_database(self, db_connection) -> bool:
        """Load game type profiles from database."""
        try:
            cursor = db_connection.execute("""
                SELECT pattern_data, game_context, confidence, frequency, success_rate
                FROM learned_patterns
                WHERE pattern_type = 'game_type_profile'
            """)
            
            for row in cursor.fetchall():
                pattern_data = json.loads(row[0])
                game_type = row[1]
                
                profile = GameTypeProfile(
                    game_type=game_type,
                    total_games=pattern_data.get('total_games', 0),
                    successful_games=pattern_data.get('successful_games', 0),
                    avg_score=pattern_data.get('avg_score', 0.0),
                    common_patterns=pattern_data.get('common_patterns', []),
                    successful_coordinates=pattern_data.get('successful_coordinates', []),
                    interactive_objects=pattern_data.get('interactive_objects', []),
                    winning_sequences=pattern_data.get('winning_sequences', []),
                    button_priorities=pattern_data.get('button_priorities', []),
                    action6_centric_count=pattern_data.get('action6_centric_count', 0)
                )
                
                self.game_type_profiles[game_type] = profile
            
            # Load button priorities from dedicated table
            cursor = db_connection.execute("""
                SELECT game_type, coordinate_x, coordinate_y, button_type, confidence, success_count,
                       score_changes, action_unlocks, test_count, last_used
                FROM button_priorities
                ORDER BY game_type, confidence DESC, success_count DESC
            """)
            
            button_priorities_by_game = {}
            for row in cursor.fetchall():
                game_type = row[0]
                if game_type not in button_priorities_by_game:
                    button_priorities_by_game[game_type] = []
                
                button_priorities_by_game[game_type].append({
                    'coordinate': (row[1], row[2]),
                    'button_type': row[3],
                    'confidence': row[4],
                    'success_count': row[5],
                    'score_changes': row[6],
                    'action_unlocks': row[7],
                    'test_count': row[8],
                    'last_used': row[9]
                })
            
            # Update profiles with button priorities
            for game_type, button_priorities in button_priorities_by_game.items():
                if game_type in self.game_type_profiles:
                    self.game_type_profiles[game_type].button_priorities = button_priorities
            
            logger.info(f"Loaded {len(self.game_type_profiles)} game type profiles and button priorities from database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load game type profiles from database: {e}")
            return False

# Global instance
game_type_classifier = GameTypeClassifier()
