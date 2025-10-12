"""
Cross-Game Transfer Learning System

Advanced system for extracting, storing, and applying transferable knowledge
across different ARC games to accelerate learning and improve performance.
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TransferablePattern:
    """Represents a pattern that can be transferred between games."""
    pattern_id: str
    pattern_type: str  # 'visual', 'spatial', 'sequence', 'strategy'
    description: str
    features: Dict[str, Any]
    success_contexts: List[Dict[str, Any]]
    failure_contexts: List[Dict[str, Any]]
    effectiveness_score: float
    confidence: float
    games_applied: Set[str]
    created_at: datetime
    last_updated: datetime

@dataclass
class GameContext:
    """Context information for pattern matching."""
    game_id: str
    grid_size: Tuple[int, int]
    color_palette: Set[int]
    object_count: int
    complexity_score: float
    action_space: List[str]
    visual_features: Dict[str, Any]

@dataclass
class TransferLearningMetrics:
    """Metrics for transfer learning effectiveness."""
    patterns_extracted: int
    patterns_applied: int
    successful_transfers: int
    failed_transfers: int
    cross_game_improvement: float
    pattern_diversity: float
    transfer_efficiency: float

class CrossGameTransferLearning:
    """
    Advanced cross-game transfer learning system that extracts generalizable
    patterns from game experiences and applies them to improve performance
    on new games.
    """

    def __init__(self, persistence_dir: str = "data/transfer_learning"):
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.transferable_patterns: Dict[str, TransferablePattern] = {}
        self.game_contexts: Dict[str, GameContext] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        self.metrics = TransferLearningMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0)

        # Configuration
        self.pattern_similarity_threshold = 0.7
        self.min_pattern_confidence = 0.6
        self.max_patterns_per_type = 100
        self.pattern_decay_days = 30

        # Load existing patterns
        self._load_patterns()

        logger.info("Cross-Game Transfer Learning System initialized")

    def extract_patterns_from_game_session(self, game_id: str, session_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract transferable patterns from a completed game session."""
        patterns = []

        try:
            # Extract visual patterns
            visual_patterns = self._extract_visual_patterns(game_id, session_data)
            patterns.extend(visual_patterns)

            # Extract spatial patterns
            spatial_patterns = self._extract_spatial_patterns(game_id, session_data)
            patterns.extend(spatial_patterns)

            # Extract sequence patterns
            sequence_patterns = self._extract_sequence_patterns(game_id, session_data)
            patterns.extend(sequence_patterns)

            # Extract strategy patterns
            strategy_patterns = self._extract_strategy_patterns(game_id, session_data)
            patterns.extend(strategy_patterns)

            # Store new patterns
            for pattern in patterns:
                self._store_pattern(pattern)

            self.metrics.patterns_extracted += len(patterns)
            logger.info(f"Extracted {len(patterns)} transferable patterns from game {game_id}")

        except Exception as e:
            logger.error(f"Error extracting patterns from game {game_id}: {e}")

        return patterns

    def extract_patterns_from_level_win(self, game_id: str, level_data: Dict[str, Any]) -> List[TransferablePattern]:
        """ENHANCED: Extract transferable patterns from individual level wins (not just full game wins)."""
        patterns = []

        try:
            # Extract patterns from level completion
            if level_data.get('level_completed', False):
                print(f"[TRANSFER] Extracting patterns from level win in game {game_id}")

                # Level-specific visual patterns
                level_visual_patterns = self._extract_level_visual_patterns(game_id, level_data)
                patterns.extend(level_visual_patterns)

                # Level completion strategies
                completion_patterns = self._extract_level_completion_patterns(game_id, level_data)
                patterns.extend(completion_patterns)

                # Winning action sequences
                winning_sequences = self._extract_winning_action_sequences(game_id, level_data)
                patterns.extend(winning_sequences)

                # Store new patterns
                for pattern in patterns:
                    self._store_pattern(pattern)

                self.metrics.patterns_extracted += len(patterns)
                logger.info(f"Extracted {len(patterns)} patterns from level win in game {game_id}")

        except Exception as e:
            logger.error(f"Error extracting level win patterns from game {game_id}: {e}")

        return patterns

    def extract_patterns_from_score_increase(self, game_id: str, score_data: Dict[str, Any]) -> List[TransferablePattern]:
        """ENHANCED: Extract transferable patterns from ANY score increases (not just high scores)."""
        patterns = []

        try:
            score_change = score_data.get('score_change', 0)
            if score_change > 0:  # Any positive score change
                print(f"[TRANSFER] Extracting patterns from +{score_change} score increase in game {game_id}")

                # Score-generating action patterns
                score_action_patterns = self._extract_score_action_patterns(game_id, score_data)
                patterns.extend(score_action_patterns)

                # Incremental progress patterns
                progress_patterns = self._extract_incremental_progress_patterns(game_id, score_data)
                patterns.extend(progress_patterns)

                # Coordinate effectiveness patterns
                coord_patterns = self._extract_coordinate_effectiveness_patterns(game_id, score_data)
                patterns.extend(coord_patterns)

                # Store new patterns
                for pattern in patterns:
                    self._store_pattern(pattern)

                self.metrics.patterns_extracted += len(patterns)
                logger.info(f"Extracted {len(patterns)} patterns from score increase (+{score_change}) in game {game_id}")

        except Exception as e:
            logger.error(f"Error extracting score increase patterns from game {game_id}: {e}")

        return patterns

    def create_abstraction_engine_patterns(self, game_context: GameContext) -> List[TransferablePattern]:
        """ENHANCED: Create abstracted meta-patterns that generalize across different game types."""
        meta_patterns = []

        try:
            print(f"[ABSTRACTION] Creating generalized patterns for game {game_context.game_id}")

            # Analyze existing patterns to find commonalities
            common_visual_elements = self._find_common_visual_elements()
            common_spatial_relationships = self._find_common_spatial_relationships()
            common_action_sequences = self._find_common_action_sequences()

            # Create meta-patterns
            if common_visual_elements:
                visual_meta_pattern = self._create_visual_meta_pattern(common_visual_elements, game_context)
                if visual_meta_pattern:
                    meta_patterns.append(visual_meta_pattern)

            if common_spatial_relationships:
                spatial_meta_pattern = self._create_spatial_meta_pattern(common_spatial_relationships, game_context)
                if spatial_meta_pattern:
                    meta_patterns.append(spatial_meta_pattern)

            if common_action_sequences:
                sequence_meta_pattern = self._create_sequence_meta_pattern(common_action_sequences, game_context)
                if sequence_meta_pattern:
                    meta_patterns.append(sequence_meta_pattern)

            # Store meta-patterns
            for pattern in meta_patterns:
                self._store_pattern(pattern)

            logger.info(f"Created {len(meta_patterns)} abstracted meta-patterns for game {game_context.game_id}")

        except Exception as e:
            logger.error(f"Error creating abstraction patterns: {e}")

        return meta_patterns

    def apply_patterns_to_new_game(self, game_context: GameContext) -> Dict[str, List[Tuple[int, int]]]:
        """ENHANCED: Apply abstracted patterns to help with new games using past success hints."""
        recommendations = {
            'high_priority_coordinates': [],
            'exploration_coordinates': [],
            'sequence_coordinates': [],
            'meta_pattern_coordinates': []
        }

        try:
            print(f"[APPLY] Applying learned patterns to new game {game_context.game_id}")

            # Get applicable patterns
            applicable_patterns = self.get_applicable_patterns(game_context)

            for pattern, similarity in applicable_patterns[:10]:  # Top 10 most similar patterns
                current_context = {
                    'current_frame': game_context.current_frame,
                    'grid_size': (len(game_context.current_frame[0]) if game_context.current_frame else 10, 
                                 len(game_context.current_frame) if game_context.current_frame else 10),
                    'reference_coordinates': [(5, 5)],  # Default reference
                    'recent_actions': game_context.recent_actions
                }

                # Apply pattern to get coordinates
                coords = self.apply_pattern_to_coordinates(pattern, current_context)

                # Categorize coordinates by pattern type and effectiveness
                if pattern.effectiveness_score > 0.8:
                    recommendations['high_priority_coordinates'].extend(coords)
                elif pattern.pattern_type == 'meta':
                    recommendations['meta_pattern_coordinates'].extend(coords)
                elif pattern.pattern_type in ['sequence', 'strategy']:
                    recommendations['sequence_coordinates'].extend(coords)
                else:
                    recommendations['exploration_coordinates'].extend(coords)

            # Remove duplicates and limit results
            for category in recommendations:
                recommendations[category] = list(set(recommendations[category]))[:5]  # Top 5 per category

            total_coords = sum(len(coords) for coords in recommendations.values())
            logger.info(f"Generated {total_coords} coordinate recommendations for game {game_context.game_id}")

        except Exception as e:
            logger.error(f"Error applying patterns to new game: {e}")

        return recommendations

    def get_applicable_patterns(self, game_context: GameContext) -> List[Tuple[TransferablePattern, float]]:
        """Find patterns that might be applicable to the given game context."""
        applicable_patterns = []

        try:
            for pattern in self.transferable_patterns.values():
                if pattern.confidence < self.min_pattern_confidence:
                    continue

                similarity = self._calculate_context_similarity(pattern, game_context)
                if similarity >= self.pattern_similarity_threshold:
                    applicable_patterns.append((pattern, similarity))

            # Sort by similarity and effectiveness
            applicable_patterns.sort(key=lambda x: (x[1], x[0].effectiveness_score), reverse=True)

            logger.debug(f"Found {len(applicable_patterns)} applicable patterns for game {game_context.game_id}")

        except Exception as e:
            logger.error(f"Error finding applicable patterns: {e}")

        return applicable_patterns

    def apply_pattern_to_coordinates(self, pattern: TransferablePattern,
                                   current_context: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Apply a transferable pattern to generate coordinate suggestions."""
        coordinates = []

        try:
            if pattern.pattern_type == 'spatial':
                coordinates = self._apply_spatial_pattern(pattern, current_context)
            elif pattern.pattern_type == 'visual':
                coordinates = self._apply_visual_pattern(pattern, current_context)
            elif pattern.pattern_type == 'sequence':
                coordinates = self._apply_sequence_pattern(pattern, current_context)
            elif pattern.pattern_type == 'strategy':
                coordinates = self._apply_strategy_pattern(pattern, current_context)

            self.metrics.patterns_applied += 1
            logger.debug(f"Applied {pattern.pattern_type} pattern {pattern.pattern_id}, generated {len(coordinates)} coordinates")

        except Exception as e:
            logger.error(f"Error applying pattern {pattern.pattern_id}: {e}")

        return coordinates

    def record_transfer_feedback(self, pattern_id: str, success: bool,
                               context: Dict[str, Any], outcome: Dict[str, Any]):
        """Record feedback on pattern transfer effectiveness."""
        try:
            pattern = self.transferable_patterns.get(pattern_id)
            if not pattern:
                return

            # Update pattern based on feedback
            if success:
                pattern.success_contexts.append({
                    **context,
                    'outcome': outcome,
                    'timestamp': datetime.now().isoformat()
                })
                pattern.effectiveness_score = min(1.0, pattern.effectiveness_score + 0.1)
                self.metrics.successful_transfers += 1
            else:
                pattern.failure_contexts.append({
                    **context,
                    'outcome': outcome,
                    'timestamp': datetime.now().isoformat()
                })
                pattern.effectiveness_score = max(0.0, pattern.effectiveness_score - 0.05)
                self.metrics.failed_transfers += 1

            # Update confidence based on total attempts
            total_attempts = len(pattern.success_contexts) + len(pattern.failure_contexts)
            if total_attempts > 0:
                success_rate = len(pattern.success_contexts) / total_attempts
                pattern.confidence = min(1.0, success_rate + (total_attempts / 100))

            pattern.last_updated = datetime.now()

            # Record transfer history
            self.transfer_history.append({
                'pattern_id': pattern_id,
                'success': success,
                'context': context,
                'outcome': outcome,
                'timestamp': datetime.now().isoformat()
            })

            self._save_patterns()

        except Exception as e:
            logger.error(f"Error recording transfer feedback: {e}")

    def _extract_visual_patterns(self, game_id: str, session_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract visual patterns from game session."""
        patterns = []

        try:
            pseudo_button_data = session_data.get('pseudo_button_learning', {})
            if not pseudo_button_data:
                return patterns

            # Group successful pseudo-buttons by visual characteristics
            successful_buttons = {}
            for coords, effect_data in pseudo_button_data.get('button_effects', {}).items():
                if effect_data.get('effectiveness_score', 0) > 0.7:
                    visual_key = self._extract_visual_signature(coords, session_data)
                    if visual_key not in successful_buttons:
                        successful_buttons[visual_key] = []
                    successful_buttons[visual_key].append((coords, effect_data))

            # Create patterns for recurring visual signatures
            for visual_signature, button_list in successful_buttons.items():
                if len(button_list) >= 2:  # Pattern needs at least 2 occurrences
                    pattern_id = self._generate_pattern_id('visual', visual_signature)

                    pattern = TransferablePattern(
                        pattern_id=pattern_id,
                        pattern_type='visual',
                        description=f"Visual pattern: {visual_signature[:50]}...",
                        features={
                            'visual_signature': visual_signature,
                            'success_rate': sum(data[1]['effectiveness_score'] for data in button_list) / len(button_list),
                            'occurrence_count': len(button_list)
                        },
                        success_contexts=[{
                            'game_id': game_id,
                            'coordinates': coords,
                            'effect_data': effect_data
                        } for coords, effect_data in button_list],
                        failure_contexts=[],
                        effectiveness_score=0.8,
                        confidence=min(1.0, len(button_list) / 5),
                        games_applied={game_id},
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )

                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error extracting visual patterns: {e}")

        return patterns

    def _extract_spatial_patterns(self, game_id: str, session_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract spatial patterns from game session."""
        patterns = []

        try:
            # Extract patterns from successful coordinate sequences
            sequences = session_data.get('pseudo_button_learning', {}).get('successful_sequences', [])

            for sequence in sequences:
                if sequence.get('score_gained', 0) > 50:  # Only high-value sequences
                    coords_list = sequence.get('sequence', [])
                    if len(coords_list) >= 2:

                        spatial_features = self._analyze_spatial_relationships(coords_list)
                        pattern_id = self._generate_pattern_id('spatial', str(spatial_features))

                        pattern = TransferablePattern(
                            pattern_id=pattern_id,
                            pattern_type='spatial',
                            description=f"Spatial sequence pattern with {len(coords_list)} coordinates",
                            features={
                                'sequence_length': len(coords_list),
                                'spatial_relationships': spatial_features,
                                'score_value': sequence.get('score_gained', 0)
                            },
                            success_contexts=[{
                                'game_id': game_id,
                                'sequence': coords_list,
                                'score_gained': sequence.get('score_gained', 0)
                            }],
                            failure_contexts=[],
                            effectiveness_score=0.7,
                            confidence=0.6,
                            games_applied={game_id},
                            created_at=datetime.now(),
                            last_updated=datetime.now()
                        )

                        patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error extracting spatial patterns: {e}")

        return patterns

    def _extract_sequence_patterns(self, game_id: str, session_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract action sequence patterns from game session."""
        patterns = []

        try:
            # Look for repeated action patterns that led to success
            action_history = session_data.get('action_history', [])
            successful_sequences = []

            # Find sequences that preceded score increases
            for i in range(len(action_history) - 2):
                sequence = action_history[i:i+3]
                if self._sequence_led_to_success(sequence, session_data):
                    successful_sequences.append(sequence)

            # Group similar sequences
            sequence_groups = self._group_similar_sequences(successful_sequences)

            for sequence_pattern, occurrences in sequence_groups.items():
                if len(occurrences) >= 2:
                    pattern_id = self._generate_pattern_id('sequence', sequence_pattern)

                    pattern = TransferablePattern(
                        pattern_id=pattern_id,
                        pattern_type='sequence',
                        description=f"Action sequence pattern: {sequence_pattern}",
                        features={
                            'sequence_pattern': sequence_pattern,
                            'occurrence_count': len(occurrences),
                            'average_success_rate': sum(occ['success_rate'] for occ in occurrences) / len(occurrences)
                        },
                        success_contexts=[{
                            'game_id': game_id,
                            'sequence': occ['sequence'],
                            'success_rate': occ['success_rate']
                        } for occ in occurrences],
                        failure_contexts=[],
                        effectiveness_score=0.6,
                        confidence=min(1.0, len(occurrences) / 3),
                        games_applied={game_id},
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )

                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error extracting sequence patterns: {e}")

        return patterns

    def _extract_strategy_patterns(self, game_id: str, session_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract high-level strategy patterns from game session."""
        patterns = []

        try:
            # Extract strategy based on overall game performance
            performance_metrics = session_data.get('performance_metrics', {})
            if performance_metrics.get('success_rate', 0) > 0.7:

                strategy_features = {
                    'action_distribution': self._analyze_action_distribution(session_data),
                    'exploration_strategy': self._analyze_exploration_pattern(session_data),
                    'timing_strategy': self._analyze_timing_patterns(session_data)
                }

                pattern_id = self._generate_pattern_id('strategy', str(strategy_features))

                pattern = TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='strategy',
                    description=f"Successful strategy pattern from {game_id}",
                    features=strategy_features,
                    success_contexts=[{
                        'game_id': game_id,
                        'performance': performance_metrics,
                        'strategy_features': strategy_features
                    }],
                    failure_contexts=[],
                    effectiveness_score=performance_metrics.get('success_rate', 0.7),
                    confidence=0.8,
                    games_applied={game_id},
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )

                patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error extracting strategy patterns: {e}")

        return patterns

    def _calculate_context_similarity(self, pattern: TransferablePattern,
                                    game_context: GameContext) -> float:
        """Calculate similarity between pattern context and current game context."""
        try:
            similarity_scores = []

            # Compare with successful contexts
            for success_context in pattern.success_contexts:
                if pattern.pattern_type == 'visual':
                    score = self._compare_visual_contexts(success_context, game_context)
                elif pattern.pattern_type == 'spatial':
                    score = self._compare_spatial_contexts(success_context, game_context)
                elif pattern.pattern_type == 'sequence':
                    score = self._compare_sequence_contexts(success_context, game_context)
                elif pattern.pattern_type == 'strategy':
                    score = self._compare_strategy_contexts(success_context, game_context)
                else:
                    score = 0.5  # Default moderate similarity

                similarity_scores.append(score)

            return max(similarity_scores) if similarity_scores else 0.0

        except Exception as e:
            logger.error(f"Error calculating context similarity: {e}")
            return 0.0

    def _apply_spatial_pattern(self, pattern: TransferablePattern,
                             current_context: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Apply spatial pattern to generate coordinates."""
        coordinates = []

        try:
            spatial_features = pattern.features.get('spatial_relationships', {})
            reference_coords = current_context.get('reference_coordinates', [(5, 5)])

            for ref_x, ref_y in reference_coords:
                # Apply spatial transformations based on pattern
                for relationship in spatial_features.get('relationships', []):
                    dx = relationship.get('dx', 0)
                    dy = relationship.get('dy', 0)
                    new_coord = (ref_x + dx, ref_y + dy)

                    # Validate coordinate is within bounds
                    grid_size = current_context.get('grid_size', (10, 10))
                    if 0 <= new_coord[0] < grid_size[0] and 0 <= new_coord[1] < grid_size[1]:
                        coordinates.append(new_coord)

        except Exception as e:
            logger.error(f"Error applying spatial pattern: {e}")

        return coordinates

    def _apply_visual_pattern(self, pattern: TransferablePattern,
                            current_context: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Apply visual pattern to generate coordinates."""
        coordinates = []

        try:
            visual_signature = pattern.features.get('visual_signature', '')
            current_frame = current_context.get('current_frame', [])

            # Find locations in current frame that match visual signature
            matches = self._find_visual_matches(visual_signature, current_frame)
            coordinates.extend(matches)

        except Exception as e:
            logger.error(f"Error applying visual pattern: {e}")

        return coordinates

    def _apply_sequence_pattern(self, pattern: TransferablePattern,
                              current_context: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Apply sequence pattern to generate coordinates."""
        coordinates = []

        try:
            sequence_pattern = pattern.features.get('sequence_pattern', '')
            last_actions = current_context.get('recent_actions', [])

            # If we're in the middle of a matching sequence, predict next coordinates
            if self._sequence_matches_pattern(last_actions, sequence_pattern):
                predicted_coords = self._predict_next_sequence_coords(sequence_pattern, current_context)
                coordinates.extend(predicted_coords)

        except Exception as e:
            logger.error(f"Error applying sequence pattern: {e}")

        return coordinates

    def _apply_strategy_pattern(self, pattern: TransferablePattern,
                              current_context: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Apply strategy pattern to generate coordinates."""
        coordinates = []

        try:
            strategy_features = pattern.features
            exploration_strategy = strategy_features.get('exploration_strategy', {})

            # Apply exploration strategy to generate coordinates
            if exploration_strategy.get('type') == 'systematic_grid':
                coordinates = self._generate_systematic_grid_coords(current_context)
            elif exploration_strategy.get('type') == 'edge_focused':
                coordinates = self._generate_edge_focused_coords(current_context)
            elif exploration_strategy.get('type') == 'center_out':
                coordinates = self._generate_center_out_coords(current_context)

        except Exception as e:
            logger.error(f"Error applying strategy pattern: {e}")

        return coordinates

    # Helper methods for pattern processing
    def _generate_pattern_id(self, pattern_type: str, features: str) -> str:
        """Generate unique ID for a pattern."""
        hash_input = f"{pattern_type}_{features}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _extract_visual_signature(self, coords: Tuple[int, int],
                                session_data: Dict[str, Any]) -> str:
        """Extract visual signature around coordinates."""
        # Simplified implementation - would need actual frame data
        return f"visual_sig_{coords[0]}_{coords[1]}"

    def _analyze_spatial_relationships(self, coords_list: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze spatial relationships in coordinate sequence."""
        relationships = []
        for i in range(len(coords_list) - 1):
            dx = coords_list[i+1][0] - coords_list[i][0]
            dy = coords_list[i+1][1] - coords_list[i][1]
            relationships.append({'dx': dx, 'dy': dy})

        return {
            'relationships': relationships,
            'pattern_type': 'sequential',
            'total_displacement': (
                coords_list[-1][0] - coords_list[0][0],
                coords_list[-1][1] - coords_list[0][1]
            )
        }

    def _store_pattern(self, pattern: TransferablePattern):
        """Store a transferable pattern."""
        self.transferable_patterns[pattern.pattern_id] = pattern
        # Automatically save patterns to disk for persistence
        self._save_patterns()

    def _save_patterns(self):
        """Save patterns to disk."""
        try:
            patterns_file = self.persistence_dir / "transferable_patterns.json"
            patterns_data = {}

            for pattern_id, pattern in self.transferable_patterns.items():
                pattern_dict = asdict(pattern)
                # Convert sets to lists for JSON serialization
                pattern_dict['games_applied'] = list(pattern_dict['games_applied'])
                pattern_dict['created_at'] = pattern_dict['created_at'].isoformat()
                pattern_dict['last_updated'] = pattern_dict['last_updated'].isoformat()
                patterns_data[pattern_id] = pattern_dict

            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving patterns: {e}")

    def _load_patterns(self):
        """Load patterns from disk."""
        try:
            patterns_file = self.persistence_dir / "transferable_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)

                for pattern_id, pattern_dict in patterns_data.items():
                    # Convert lists back to sets and strings back to datetime
                    pattern_dict['games_applied'] = set(pattern_dict['games_applied'])
                    pattern_dict['created_at'] = datetime.fromisoformat(pattern_dict['created_at'])
                    pattern_dict['last_updated'] = datetime.fromisoformat(pattern_dict['last_updated'])

                    pattern = TransferablePattern(**pattern_dict)
                    self.transferable_patterns[pattern_id] = pattern

                logger.info(f"Loaded {len(self.transferable_patterns)} transferable patterns")

        except Exception as e:
            logger.error(f"Error loading patterns: {e}")

    def get_transfer_learning_metrics(self) -> TransferLearningMetrics:
        """Get current transfer learning metrics."""
        # Update pattern diversity
        pattern_types = set(p.pattern_type for p in self.transferable_patterns.values())
        self.metrics.pattern_diversity = len(pattern_types) / 4.0  # 4 pattern types

        # Update transfer efficiency
        total_transfers = self.metrics.successful_transfers + self.metrics.failed_transfers
        if total_transfers > 0:
            self.metrics.transfer_efficiency = self.metrics.successful_transfers / total_transfers

        return self.metrics

    def cleanup_old_patterns(self):
        """Remove old or ineffective patterns."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.pattern_decay_days)
            patterns_to_remove = []

            for pattern_id, pattern in self.transferable_patterns.items():
                # Remove old patterns with low effectiveness
                if (pattern.last_updated < cutoff_date and
                    pattern.effectiveness_score < 0.3):
                    patterns_to_remove.append(pattern_id)

            for pattern_id in patterns_to_remove:
                del self.transferable_patterns[pattern_id]

            if patterns_to_remove:
                logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")
                self._save_patterns()

        except Exception as e:
            logger.error(f"Error cleaning up patterns: {e}")

    # Placeholder implementations for helper methods
    def _sequence_led_to_success(self, sequence, session_data): return True
    def _group_similar_sequences(self, sequences): return {}
    def _analyze_action_distribution(self, session_data): return {}
    def _analyze_exploration_pattern(self, session_data): return {'type': 'systematic_grid'}
    def _analyze_timing_patterns(self, session_data): return {}
    def _compare_visual_contexts(self, context1, context2): return 0.8
    def _compare_spatial_contexts(self, context1, context2): return 0.8
    def _compare_sequence_contexts(self, context1, context2): return 0.8
    def _compare_strategy_contexts(self, context1, context2): return 0.8
    def _find_visual_matches(self, signature, frame): return []
    def _sequence_matches_pattern(self, actions, pattern): return True
    def _predict_next_sequence_coords(self, pattern, context): return []
    def _generate_systematic_grid_coords(self, context): return []
    def _generate_edge_focused_coords(self, context): return []
    def _generate_center_out_coords(self, context): return []

    # ENHANCED: Level Win Pattern Extraction Methods
    def _extract_level_visual_patterns(self, game_id: str, level_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract visual patterns specific to level completion."""
        patterns = []
        
        try:
            # Extract visual elements that led to level completion
            winning_frame = level_data.get('final_frame', [])
            winning_coords = level_data.get('winning_coordinates', [])
            
            if winning_frame and winning_coords:
                for coords in winning_coords:
                    visual_signature = self._extract_visual_signature_enhanced(coords, winning_frame)
                    
                    pattern_id = self._generate_pattern_id('level_visual', f"{visual_signature}_{game_id}")
                    
                    pattern = TransferablePattern(
                        pattern_id=pattern_id,
                        pattern_type='level_visual',
                        description=f"Level completion visual pattern from {game_id}",
                        features={
                            'visual_signature': visual_signature,
                            'completion_coordinates': coords,
                            'level_context': level_data.get('level_context', {}),
                            'success_type': 'level_completion'
                        },
                        success_contexts=[{
                            'game_id': game_id,
                            'level_id': level_data.get('level_id', 'unknown'),
                            'completion_method': 'visual_pattern',
                            'effectiveness': level_data.get('completion_score', 1.0)
                        }],
                        failure_contexts=[],
                        effectiveness_score=0.9,  # High score for level completions
                        confidence=0.8,
                        games_applied={game_id},
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.error(f"Error extracting level visual patterns: {e}")
            
        return patterns

    def _extract_level_completion_patterns(self, game_id: str, level_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract strategy patterns that led to level completion."""
        patterns = []
        
        try:
            actions_to_completion = level_data.get('actions_to_completion', [])
            completion_strategy = level_data.get('strategy_used', 'unknown')
            
            if len(actions_to_completion) >= 3:  # Need meaningful sequence
                pattern_id = self._generate_pattern_id('level_strategy', f"{completion_strategy}_{game_id}")
                
                pattern = TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='level_strategy',
                    description=f"Level completion strategy: {completion_strategy}",
                    features={
                        'completion_sequence': actions_to_completion,
                        'strategy_type': completion_strategy,
                        'sequence_length': len(actions_to_completion),
                        'completion_efficiency': level_data.get('efficiency_score', 0.7)
                    },
                    success_contexts=[{
                        'game_id': game_id,
                        'level_id': level_data.get('level_id', 'unknown'),
                        'completion_actions': actions_to_completion,
                        'strategy': completion_strategy
                    }],
                    failure_contexts=[],
                    effectiveness_score=0.85,
                    confidence=0.75,
                    games_applied={game_id},
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error extracting level completion patterns: {e}")
            
        return patterns

    def _extract_winning_action_sequences(self, game_id: str, level_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract action sequences that directly led to level wins."""
        patterns = []
        
        try:
            pre_win_actions = level_data.get('pre_completion_sequence', [])
            
            if len(pre_win_actions) >= 2:
                # Create pattern for the sequence that led to completion
                sequence_str = "_".join(str(action) for action in pre_win_actions[-5:])  # Last 5 actions
                pattern_id = self._generate_pattern_id('winning_sequence', sequence_str)
                
                pattern = TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='winning_sequence',
                    description=f"Action sequence leading to level completion",
                    features={
                        'winning_sequence': pre_win_actions[-5:],
                        'sequence_pattern': sequence_str,
                        'win_probability': 0.9,
                        'context_type': 'level_completion'
                    },
                    success_contexts=[{
                        'game_id': game_id,
                        'sequence': pre_win_actions,
                        'outcome': 'level_completed',
                        'timestamp': datetime.now().isoformat()
                    }],
                    failure_contexts=[],
                    effectiveness_score=0.95,  # Very high for winning sequences
                    confidence=0.9,
                    games_applied={game_id},
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error extracting winning action sequences: {e}")
            
        return patterns

    # ENHANCED: Score Increase Pattern Extraction Methods
    def _extract_score_action_patterns(self, game_id: str, score_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract patterns from actions that generated score increases."""
        patterns = []
        
        try:
            scoring_action = score_data.get('scoring_action', {})
            score_change = score_data.get('score_change', 0)
            action_context = score_data.get('action_context', {})
            
            if scoring_action and score_change > 0:
                action_type = scoring_action.get('action_type', 'unknown')
                coordinates = scoring_action.get('coordinates', None)
                
                pattern_id = self._generate_pattern_id('score_action', f"{action_type}_{coordinates}_{score_change}")
                
                pattern = TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='score_action',
                    description=f"Score-generating action: {action_type} (+{score_change})",
                    features={
                        'action_type': action_type,
                        'coordinates': coordinates,
                        'score_value': score_change,
                        'context_features': action_context,
                        'efficiency': score_change / max(1, action_context.get('actions_taken', 1))
                    },
                    success_contexts=[{
                        'game_id': game_id,
                        'action': scoring_action,
                        'score_gained': score_change,
                        'context': action_context
                    }],
                    failure_contexts=[],
                    effectiveness_score=min(1.0, score_change / 100.0),  # Scale based on score
                    confidence=0.7,
                    games_applied={game_id},
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error extracting score action patterns: {e}")
            
        return patterns

    def _extract_incremental_progress_patterns(self, game_id: str, score_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract patterns from incremental progress and small wins."""
        patterns = []
        
        try:
            progress_sequence = score_data.get('progress_sequence', [])
            cumulative_improvement = score_data.get('cumulative_score', 0)
            
            if len(progress_sequence) >= 2 and cumulative_improvement > 0:
                pattern_id = self._generate_pattern_id('incremental_progress', f"{game_id}_{cumulative_improvement}")
                
                pattern = TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='incremental_progress',
                    description=f"Incremental progress pattern (+{cumulative_improvement} total)",
                    features={
                        'progress_steps': progress_sequence,
                        'total_improvement': cumulative_improvement,
                        'progress_rate': cumulative_improvement / len(progress_sequence),
                        'consistency': self._calculate_progress_consistency(progress_sequence)
                    },
                    success_contexts=[{
                        'game_id': game_id,
                        'progress_sequence': progress_sequence,
                        'improvement': cumulative_improvement,
                        'step_count': len(progress_sequence)
                    }],
                    failure_contexts=[],
                    effectiveness_score=min(1.0, cumulative_improvement / 50.0),
                    confidence=0.6,
                    games_applied={game_id},
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error extracting incremental progress patterns: {e}")
            
        return patterns

    def _extract_coordinate_effectiveness_patterns(self, game_id: str, score_data: Dict[str, Any]) -> List[TransferablePattern]:
        """Extract patterns about which coordinates are most effective for scoring."""
        patterns = []
        
        try:
            effective_coordinates = score_data.get('effective_coordinates', [])
            coordinate_scores = score_data.get('coordinate_scores', {})
            
            if effective_coordinates:
                # Group coordinates by effectiveness
                high_value_coords = [(coord, score) for coord, score in coordinate_scores.items() if score > 20]
                
                if high_value_coords:
                    pattern_id = self._generate_pattern_id('coord_effectiveness', f"{game_id}_high_value")
                    
                    pattern = TransferablePattern(
                        pattern_id=pattern_id,
                        pattern_type='coord_effectiveness',
                        description=f"High-value coordinate pattern",
                        features={
                            'high_value_coordinates': high_value_coords,
                            'average_value': sum(score for _, score in high_value_coords) / len(high_value_coords),
                            'coordinate_distribution': self._analyze_coordinate_distribution(high_value_coords),
                            'spatial_clusters': self._find_coordinate_clusters(high_value_coords)
                        },
                        success_contexts=[{
                            'game_id': game_id,
                            'coordinates': effective_coordinates,
                            'scores': coordinate_scores
                        }],
                        failure_contexts=[],
                        effectiveness_score=0.8,
                        confidence=0.75,
                        games_applied={game_id},
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.error(f"Error extracting coordinate effectiveness patterns: {e}")
            
        return patterns

    # ENHANCED: Abstraction Engine Methods
    def _find_common_visual_elements(self) -> Dict[str, Any]:
        """Find visual elements that appear across multiple successful patterns."""
        common_elements = {}
        
        try:
            visual_patterns = [p for p in self.transferable_patterns.values() 
                             if p.pattern_type in ['visual', 'level_visual']]
            
            if len(visual_patterns) >= 2:
                # Analyze visual signatures for commonalities
                all_signatures = []
                for pattern in visual_patterns:
                    signature = pattern.features.get('visual_signature', '')
                    if signature:
                        all_signatures.append(signature)
                
                # Find recurring visual elements
                common_elements = {
                    'recurring_signatures': self._find_recurring_elements(all_signatures),
                    'pattern_count': len(visual_patterns),
                    'success_rate': sum(p.effectiveness_score for p in visual_patterns) / len(visual_patterns)
                }
                
        except Exception as e:
            logger.error(f"Error finding common visual elements: {e}")
            
        return common_elements

    def _find_common_spatial_relationships(self) -> Dict[str, Any]:
        """Find spatial relationships that work across multiple games."""
        common_relationships = {}
        
        try:
            spatial_patterns = [p for p in self.transferable_patterns.values() 
                              if p.pattern_type in ['spatial', 'coord_effectiveness']]
            
            if len(spatial_patterns) >= 2:
                all_relationships = []
                for pattern in spatial_patterns:
                    relationships = pattern.features.get('spatial_relationships', {})
                    if relationships:
                        all_relationships.append(relationships)
                
                common_relationships = {
                    'common_displacements': self._find_common_displacements(all_relationships),
                    'pattern_count': len(spatial_patterns),
                    'effectiveness': sum(p.effectiveness_score for p in spatial_patterns) / len(spatial_patterns)
                }
                
        except Exception as e:
            logger.error(f"Error finding common spatial relationships: {e}")
            
        return common_relationships

    def _find_common_action_sequences(self) -> Dict[str, Any]:
        """Find action sequences that succeed across multiple contexts."""
        common_sequences = {}
        
        try:
            sequence_patterns = [p for p in self.transferable_patterns.values() 
                               if p.pattern_type in ['sequence', 'winning_sequence', 'score_action']]
            
            if len(sequence_patterns) >= 2:
                all_sequences = []
                for pattern in sequence_patterns:
                    sequence = pattern.features.get('sequence_pattern', '') or pattern.features.get('winning_sequence', [])
                    if sequence:
                        all_sequences.append(sequence)
                
                common_sequences = {
                    'recurring_sequences': self._find_recurring_sequences(all_sequences),
                    'pattern_count': len(sequence_patterns),
                    'success_rate': sum(p.effectiveness_score for p in sequence_patterns) / len(sequence_patterns)
                }
                
        except Exception as e:
            logger.error(f"Error finding common action sequences: {e}")
            
        return common_sequences

    # ENHANCED: Meta-Pattern Creation Methods
    def _create_visual_meta_pattern(self, common_elements: Dict[str, Any], game_context: GameContext) -> TransferablePattern:
        """Create a meta-pattern from common visual elements."""
        try:
            if common_elements.get('pattern_count', 0) >= 2:
                pattern_id = self._generate_pattern_id('meta_visual', str(common_elements))
                
                return TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='meta_visual',
                    description=f"Meta visual pattern from {common_elements['pattern_count']} games",
                    features={
                        'meta_signatures': common_elements.get('recurring_signatures', []),
                        'generalization_level': 'cross_game',
                        'source_pattern_count': common_elements['pattern_count'],
                        'meta_success_rate': common_elements.get('success_rate', 0.7)
                    },
                    success_contexts=[{
                        'pattern_type': 'meta_aggregation',
                        'source_patterns': common_elements['pattern_count'],
                        'context': 'visual_abstraction'
                    }],
                    failure_contexts=[],
                    effectiveness_score=common_elements.get('success_rate', 0.7),
                    confidence=0.8,
                    games_applied=set(),
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error creating visual meta-pattern: {e}")
        return None

    def _create_spatial_meta_pattern(self, common_relationships: Dict[str, Any], game_context: GameContext) -> TransferablePattern:
        """Create a meta-pattern from common spatial relationships."""
        try:
            if common_relationships.get('pattern_count', 0) >= 2:
                pattern_id = self._generate_pattern_id('meta_spatial', str(common_relationships))
                
                return TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='meta_spatial',
                    description=f"Meta spatial pattern from {common_relationships['pattern_count']} games",
                    features={
                        'meta_displacements': common_relationships.get('common_displacements', []),
                        'generalization_level': 'cross_game',
                        'source_pattern_count': common_relationships['pattern_count'],
                        'meta_effectiveness': common_relationships.get('effectiveness', 0.7)
                    },
                    success_contexts=[{
                        'pattern_type': 'meta_aggregation',
                        'source_patterns': common_relationships['pattern_count'],
                        'context': 'spatial_abstraction'
                    }],
                    failure_contexts=[],
                    effectiveness_score=common_relationships.get('effectiveness', 0.7),
                    confidence=0.8,
                    games_applied=set(),
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error creating spatial meta-pattern: {e}")
        return None

    def _create_sequence_meta_pattern(self, common_sequences: Dict[str, Any], game_context: GameContext) -> TransferablePattern:
        """Create a meta-pattern from common action sequences."""
        try:
            if common_sequences.get('pattern_count', 0) >= 2:
                pattern_id = self._generate_pattern_id('meta_sequence', str(common_sequences))
                
                return TransferablePattern(
                    pattern_id=pattern_id,
                    pattern_type='meta_sequence',
                    description=f"Meta sequence pattern from {common_sequences['pattern_count']} games",
                    features={
                        'meta_sequences': common_sequences.get('recurring_sequences', []),
                        'generalization_level': 'cross_game',
                        'source_pattern_count': common_sequences['pattern_count'],
                        'meta_success_rate': common_sequences.get('success_rate', 0.7)
                    },
                    success_contexts=[{
                        'pattern_type': 'meta_aggregation',
                        'source_patterns': common_sequences['pattern_count'],
                        'context': 'sequence_abstraction'
                    }],
                    failure_contexts=[],
                    effectiveness_score=common_sequences.get('success_rate', 0.7),
                    confidence=0.8,
                    games_applied=set(),
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error creating sequence meta-pattern: {e}")
        return None

    # ENHANCED: Utility Helper Methods
    def _extract_visual_signature_enhanced(self, coords: Tuple[int, int], frame: List[List[int]]) -> str:
        """Enhanced visual signature extraction with surrounding context."""
        try:
            if not frame or not coords:
                return "empty_signature"
            
            x, y = coords
            signature_parts = []
            
            # Extract 3x3 region around coordinates
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= ny < len(frame) and 0 <= nx < len(frame[0]):
                        cell_value = frame[ny][nx]
                        signature_parts.append(str(cell_value))
                    else:
                        signature_parts.append("X")  # Out of bounds marker
            
            return "_".join(signature_parts)
        except Exception as e:
            logger.error(f"Error extracting enhanced visual signature: {e}")
            return "error_signature"

    def _calculate_progress_consistency(self, progress_sequence: List[float]) -> float:
        """Calculate how consistent the progress improvements are."""
        try:
            if len(progress_sequence) < 2:
                return 0.0
            
            # Calculate variance in progress steps
            differences = [progress_sequence[i+1] - progress_sequence[i] for i in range(len(progress_sequence)-1)]
            positive_diffs = [d for d in differences if d > 0]
            
            if not positive_diffs:
                return 0.0
            
            # Higher consistency = lower variance in positive improvements
            avg_improvement = sum(positive_diffs) / len(positive_diffs)
            variance = sum((d - avg_improvement) ** 2 for d in positive_diffs) / len(positive_diffs)
            
            # Convert to consistency score (0-1, higher = more consistent)
            consistency = max(0.0, 1.0 - (variance / (avg_improvement + 1)))
            return min(1.0, consistency)
            
        except Exception as e:
            logger.error(f"Error calculating progress consistency: {e}")
            return 0.5

    def _analyze_coordinate_distribution(self, coord_score_pairs: List[Tuple[Tuple[int, int], float]]) -> Dict[str, Any]:
        """Analyze the distribution of effective coordinates."""
        try:
            if not coord_score_pairs:
                return {}
            
            coords = [coord for coord, _ in coord_score_pairs]
            scores = [score for _, score in coord_score_pairs]
            
            # Calculate center of mass
            center_x = sum(coord[0] for coord in coords) / len(coords)
            center_y = sum(coord[1] for coord in coords) / len(coords)
            
            # Calculate spread
            spread_x = max(coord[0] for coord in coords) - min(coord[0] for coord in coords)
            spread_y = max(coord[1] for coord in coords) - min(coord[1] for coord in coords)
            
            return {
                'center_of_mass': (center_x, center_y),
                'spread': (spread_x, spread_y),
                'density': len(coords) / max(1, spread_x * spread_y),
                'score_range': (min(scores), max(scores)),
                'average_score': sum(scores) / len(scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing coordinate distribution: {e}")
            return {}

    def _find_coordinate_clusters(self, coord_score_pairs: List[Tuple[Tuple[int, int], float]]) -> List[Dict[str, Any]]:
        """Find clusters of high-value coordinates."""
        clusters = []
        
        try:
            if len(coord_score_pairs) < 2:
                return clusters
            
            # Simple clustering: group coordinates within distance threshold
            cluster_distance = 5
            used_coords = set()
            
            for coord, score in coord_score_pairs:
                if coord in used_coords:
                    continue
                
                # Start new cluster
                cluster = {
                    'center': coord,
                    'coordinates': [coord],
                    'scores': [score],
                    'avg_score': score
                }
                used_coords.add(coord)
                
                # Find nearby coordinates
                for other_coord, other_score in coord_score_pairs:
                    if other_coord in used_coords:
                        continue
                    
                    distance = ((coord[0] - other_coord[0]) ** 2 + (coord[1] - other_coord[1]) ** 2) ** 0.5
                    if distance <= cluster_distance:
                        cluster['coordinates'].append(other_coord)
                        cluster['scores'].append(other_score)
                        used_coords.add(other_coord)
                
                # Update cluster statistics
                if len(cluster['coordinates']) > 1:
                    cluster['avg_score'] = sum(cluster['scores']) / len(cluster['scores'])
                    # Recalculate center
                    cluster['center'] = (
                        sum(c[0] for c in cluster['coordinates']) / len(cluster['coordinates']),
                        sum(c[1] for c in cluster['coordinates']) / len(cluster['coordinates'])
                    )
                    clusters.append(cluster)
            
        except Exception as e:
            logger.error(f"Error finding coordinate clusters: {e}")
            
        return clusters

    def _find_recurring_elements(self, elements: List[str]) -> List[str]:
        """Find elements that recur across multiple contexts."""
        try:
            if len(elements) < 2:
                return []
            
            # Count occurrences of each element
            element_counts = {}
            for element in elements:
                element_counts[element] = element_counts.get(element, 0) + 1
            
            # Return elements that appear in multiple contexts
            recurring = [element for element, count in element_counts.items() if count >= 2]
            return recurring[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error finding recurring elements: {e}")
            return []

    def _find_common_displacements(self, relationships_list: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """Find displacement patterns that recur across spatial relationships."""
        try:
            all_displacements = []
            
            for relationships in relationships_list:
                for rel in relationships.get('relationships', []):
                    dx = rel.get('dx', 0)
                    dy = rel.get('dy', 0)
                    all_displacements.append({'dx': dx, 'dy': dy})
            
            # Count displacement patterns
            displacement_counts = {}
            for disp in all_displacements:
                key = f"{disp['dx']},{disp['dy']}"
                displacement_counts[key] = displacement_counts.get(key, 0) + 1
            
            # Return displacements that appear multiple times
            common_displacements = []
            for disp_str, count in displacement_counts.items():
                if count >= 2:
                    dx, dy = map(int, disp_str.split(','))
                    common_displacements.append({'dx': dx, 'dy': dy, 'frequency': count})
            
            return common_displacements[:5]  # Top 5 most common
            
        except Exception as e:
            logger.error(f"Error finding common displacements: {e}")
            return []

    def _find_recurring_sequences(self, sequences: List) -> List:
        """Find action sequences that recur across different contexts."""
        try:
            sequence_counts = {}
            
            for sequence in sequences:
                if isinstance(sequence, list):
                    seq_str = "_".join(map(str, sequence))
                else:
                    seq_str = str(sequence)
                
                sequence_counts[seq_str] = sequence_counts.get(seq_str, 0) + 1
            
            # Return sequences that appear multiple times
            recurring = []
            for seq_str, count in sequence_counts.items():
                if count >= 2:
                    recurring.append({
                        'sequence': seq_str,
                        'frequency': count,
                        'pattern': seq_str.split('_') if '_' in seq_str else [seq_str]
                    })
            
            return recurring[:5]  # Top 5 most recurring
            
        except Exception as e:
            logger.error(f"Error finding recurring sequences: {e}")
            return []


# Global instance
_transfer_learning_system = None

def get_transfer_learning_system() -> CrossGameTransferLearning:
    """Get the global transfer learning system instance."""
    global _transfer_learning_system
    if _transfer_learning_system is None:
        _transfer_learning_system = CrossGameTransferLearning()
    return _transfer_learning_system