"""
Knowledge Transfer

Handles knowledge transfer between different games and training sessions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class KnowledgeTransfer:
    """Manages knowledge transfer between games and sessions."""
    
    def __init__(self, transfer_threshold: float = 0.6):
        self.transfer_threshold = transfer_threshold
        self.knowledge_base = {}
        self.transfer_history = []
        self.transfer_effectiveness = defaultdict(list)
        self.game_similarities = {}
    
    def transfer_knowledge(self, source_game: str, target_game: str, 
                          knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from source game to target game."""
        try:
            transfer_result = {
                'timestamp': datetime.now(),
                'source_game': source_game,
                'target_game': target_game,
                'knowledge_type': knowledge.get('type', 'unknown'),
                'transferred_items': [],
                'effectiveness': 0.0,
                'confidence': 0.0
            }
            
            # Calculate game similarity
            similarity = self._calculate_game_similarity(source_game, target_game)
            transfer_result['game_similarity'] = similarity
            
            if similarity < self.transfer_threshold:
                transfer_result['status'] = 'skipped'
                transfer_result['reason'] = f'Game similarity {similarity:.2f} below threshold {self.transfer_threshold}'
                logger.info(f"Skipped knowledge transfer: low similarity {similarity:.2f}")
                return transfer_result
            
            # Transfer applicable knowledge
            transferred_items = self._transfer_applicable_knowledge(source_game, target_game, knowledge)
            transfer_result['transferred_items'] = transferred_items
            
            # Calculate transfer effectiveness
            effectiveness = self._calculate_transfer_effectiveness(transferred_items, knowledge)
            transfer_result['effectiveness'] = effectiveness
            
            # Calculate transfer confidence
            confidence = self._calculate_transfer_confidence(similarity, effectiveness, transferred_items)
            transfer_result['confidence'] = confidence
            
            # Update knowledge base
            self._update_knowledge_base(target_game, transferred_items)
            
            # Record transfer
            transfer_result['status'] = 'success'
            self.transfer_history.append(transfer_result)
            self.transfer_effectiveness[f"{source_game}->{target_game}"].append(effectiveness)
            
            logger.info(f"Transferred {len(transferred_items)} knowledge items from {source_game} to {target_game}")
            return transfer_result
            
        except Exception as e:
            logger.error(f"Error transferring knowledge: {e}")
            return {
                'timestamp': datetime.now(),
                'source_game': source_game,
                'target_game': target_game,
                'status': 'error',
                'error': str(e),
                'transferred_items': [],
                'effectiveness': 0.0,
                'confidence': 0.0
            }
    
    def get_transferable_knowledge(self, source_game: str, target_game: str) -> List[Dict[str, Any]]:
        """Get knowledge that can be transferred between games."""
        try:
            if source_game not in self.knowledge_base:
                return []
            
            source_knowledge = self.knowledge_base[source_game]
            transferable = []
            
            for knowledge_item in source_knowledge:
                # Check if knowledge is transferable
                if self._is_knowledge_transferable(knowledge_item, target_game):
                    transferable.append(knowledge_item)
            
            return transferable
            
        except Exception as e:
            logger.error(f"Error getting transferable knowledge: {e}")
            return []
    
    def _calculate_game_similarity(self, game1: str, game2: str) -> float:
        """Calculate similarity between two games."""
        try:
            # Check if we have cached similarity
            cache_key = f"{game1}->{game2}"
            if cache_key in self.game_similarities:
                return self.game_similarities[cache_key]
            
            # Calculate similarity based on available knowledge
            knowledge1 = self.knowledge_base.get(game1, [])
            knowledge2 = self.knowledge_base.get(game2, [])
            
            if not knowledge1 or not knowledge2:
                similarity = 0.5  # Default similarity
            else:
                # Calculate similarity based on knowledge types
                types1 = set(item.get('type', 'unknown') for item in knowledge1)
                types2 = set(item.get('type', 'unknown') for item in knowledge2)
                
                if types1 and types2:
                    intersection = len(types1.intersection(types2))
                    union = len(types1.union(types2))
                    similarity = intersection / union if union > 0 else 0.0
                else:
                    similarity = 0.0
            
            # Cache the similarity
            self.game_similarities[cache_key] = similarity
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating game similarity: {e}")
            return 0.0
    
    def _transfer_applicable_knowledge(self, source_game: str, target_game: str, 
                                     knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transfer knowledge that is applicable to the target game."""
        try:
            transferred_items = []
            
            # Transfer action patterns
            if 'action_patterns' in knowledge:
                action_patterns = knowledge['action_patterns']
                for pattern in action_patterns:
                    if self._is_action_pattern_transferable(pattern, target_game):
                        transferred_items.append({
                            'type': 'action_pattern',
                            'pattern': pattern,
                            'source': source_game,
                            'confidence': pattern.get('confidence', 0.5)
                        })
            
            # Transfer coordinate patterns
            if 'coordinate_patterns' in knowledge:
                coordinate_patterns = knowledge['coordinate_patterns']
                for pattern in coordinate_patterns:
                    if self._is_coordinate_pattern_transferable(pattern, target_game):
                        transferred_items.append({
                            'type': 'coordinate_pattern',
                            'pattern': pattern,
                            'source': source_game,
                            'confidence': pattern.get('confidence', 0.5)
                        })
            
            # Transfer learning strategies
            if 'learning_strategies' in knowledge:
                strategies = knowledge['learning_strategies']
                for strategy in strategies:
                    if self._is_strategy_transferable(strategy, target_game):
                        transferred_items.append({
                            'type': 'learning_strategy',
                            'strategy': strategy,
                            'source': source_game,
                            'confidence': strategy.get('confidence', 0.5)
                        })
            
            return transferred_items
            
        except Exception as e:
            logger.error(f"Error transferring applicable knowledge: {e}")
            return []
    
    def _is_knowledge_transferable(self, knowledge_item: Dict[str, Any], target_game: str) -> bool:
        """Check if a knowledge item is transferable to the target game."""
        try:
            knowledge_type = knowledge_item.get('type', 'unknown')
            
            # Check transferability based on type
            if knowledge_type == 'action_pattern':
                return self._is_action_pattern_transferable(knowledge_item, target_game)
            elif knowledge_type == 'coordinate_pattern':
                return self._is_coordinate_pattern_transferable(knowledge_item, target_game)
            elif knowledge_type == 'learning_strategy':
                return self._is_strategy_transferable(knowledge_item, target_game)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error checking knowledge transferability: {e}")
            return False
    
    def _is_action_pattern_transferable(self, pattern: Dict[str, Any], target_game: str) -> bool:
        """Check if an action pattern is transferable."""
        try:
            # Check pattern confidence
            confidence = pattern.get('confidence', 0.0)
            if confidence < 0.5:
                return False
            
            # Check pattern effectiveness
            effectiveness = pattern.get('effectiveness', 0.0)
            if effectiveness < 0.3:
                return False
            
            # Check if pattern is game-agnostic
            game_specific = pattern.get('game_specific', False)
            if game_specific:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking action pattern transferability: {e}")
            return False
    
    def _is_coordinate_pattern_transferable(self, pattern: Dict[str, Any], target_game: str) -> bool:
        """Check if a coordinate pattern is transferable."""
        try:
            # Check pattern confidence
            confidence = pattern.get('confidence', 0.0)
            if confidence < 0.4:
                return False
            
            # Check if pattern is relative (not absolute coordinates)
            is_relative = pattern.get('is_relative', False)
            if not is_relative:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking coordinate pattern transferability: {e}")
            return False
    
    def _is_strategy_transferable(self, strategy: Dict[str, Any], target_game: str) -> bool:
        """Check if a learning strategy is transferable."""
        try:
            # Check strategy confidence
            confidence = strategy.get('confidence', 0.0)
            if confidence < 0.6:
                return False
            
            # Check if strategy is general (not game-specific)
            is_general = strategy.get('is_general', True)
            if not is_general:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking strategy transferability: {e}")
            return False
    
    def _calculate_transfer_effectiveness(self, transferred_items: List[Dict[str, Any]], 
                                        original_knowledge: Dict[str, Any]) -> float:
        """Calculate effectiveness of knowledge transfer."""
        try:
            if not transferred_items:
                return 0.0
            
            # Calculate average confidence of transferred items
            confidences = [item.get('confidence', 0.0) for item in transferred_items]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate transfer ratio
            total_knowledge_items = len(original_knowledge.get('items', []))
            transfer_ratio = len(transferred_items) / max(total_knowledge_items, 1)
            
            # Combine confidence and transfer ratio
            effectiveness = (avg_confidence + transfer_ratio) / 2.0
            
            return min(1.0, max(0.0, effectiveness))
            
        except Exception as e:
            logger.error(f"Error calculating transfer effectiveness: {e}")
            return 0.0
    
    def _calculate_transfer_confidence(self, similarity: float, effectiveness: float, 
                                     transferred_items: List[Dict[str, Any]]) -> float:
        """Calculate confidence in knowledge transfer."""
        try:
            # Base confidence on similarity and effectiveness
            base_confidence = (similarity + effectiveness) / 2.0
            
            # Adjust based on number of transferred items
            item_count = len(transferred_items)
            if item_count > 5:
                base_confidence += 0.1
            elif item_count > 2:
                base_confidence += 0.05
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating transfer confidence: {e}")
            return 0.0
    
    def _update_knowledge_base(self, game: str, transferred_items: List[Dict[str, Any]]) -> None:
        """Update knowledge base with transferred items."""
        try:
            if game not in self.knowledge_base:
                self.knowledge_base[game] = []
            
            # Add transferred items to knowledge base
            for item in transferred_items:
                item['transferred_at'] = datetime.now()
                self.knowledge_base[game].append(item)
            
            # Keep only recent knowledge (limit to 1000 items per game)
            if len(self.knowledge_base[game]) > 1000:
                self.knowledge_base[game] = self.knowledge_base[game][-500:]
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get knowledge transfer statistics."""
        try:
            total_transfers = len(self.transfer_history)
            successful_transfers = sum(1 for t in self.transfer_history if t.get('status') == 'success')
            
            # Calculate average effectiveness
            all_effectiveness = []
            for transfer in self.transfer_history:
                if 'effectiveness' in transfer:
                    all_effectiveness.append(transfer['effectiveness'])
            
            avg_effectiveness = sum(all_effectiveness) / len(all_effectiveness) if all_effectiveness else 0.0
            
            return {
                'total_transfers': total_transfers,
                'successful_transfers': successful_transfers,
                'success_rate': successful_transfers / max(total_transfers, 1),
                'average_effectiveness': avg_effectiveness,
                'games_with_knowledge': len(self.knowledge_base),
                'total_knowledge_items': sum(len(items) for items in self.knowledge_base.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting transfer statistics: {e}")
            return {}
    
    def reset_knowledge_transfer(self) -> None:
        """Reset knowledge transfer state."""
        self.knowledge_base.clear()
        self.transfer_history.clear()
        self.transfer_effectiveness.clear()
        self.game_similarities.clear()
        logger.info("Knowledge transfer reset")
