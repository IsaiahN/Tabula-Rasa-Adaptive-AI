"""
Enhanced Action 6 Coordinator

Provides advanced coordinate selection for Action 6 using pseudo-button detection
and effectiveness testing. Integrates with the existing vision system and database
for learning and improvement.

Features:
- Pseudo-button detection using computer vision
- Frame difference analysis for effectiveness testing
- Integration with coordinate intelligence database
- Learning from click results to improve future selections
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import random
import time
from datetime import datetime

from ..core.cross_game_transfer_learning import get_transfer_learning_system, GameContext

logger = logging.getLogger(__name__)


class Action6Coordinator:
    """Enhanced coordinator for Action 6 with pseudo-button detection."""

    def __init__(self, db_interface: Optional[Any] = None, vision_detector: Optional[Any] = None):
        """Initialize the Action 6 coordinator.

        Args:
            db_interface: Database interface for storing learning data
            vision_detector: Pseudo-button detector instance
        """
        self.db_interface = db_interface
        self.vision_detector = vision_detector
        self.stats = {
            'action6_selections': 0,
            'button_based_selections': 0,
            'successful_clicks': 0,
            'failed_clicks': 0,
            'penalty_avoidances': 0,  # NEW: Track penalty-based avoidances
            'penalty_recoveries': 0   # NEW: Track successful recoveries
        }

        # Pseudo-button learning state per game session
        self.game_sessions = {}  # game_id -> session learning data
        
        # ENHANCED: Initialize penalty decay system
        self.penalty_system = None
        self._initialize_penalty_system()

        # ENHANCED: Initialize cross-game transfer learning system
        self.transfer_learning = get_transfer_learning_system()

    def _initialize_penalty_system(self) -> None:
        """Initialize the penalty decay system for enhanced coordinate selection."""
        try:
            from src.core.penalty_decay_system import get_penalty_decay_system
            self.penalty_system = get_penalty_decay_system(self.db_interface)
            logger.debug("Penalty decay system initialized in Action6Coordinator")
        except Exception as e:
            logger.warning(f"Could not initialize penalty decay system: {e}")
            self.penalty_system = None

    async def _ensure_penalty_system_ready(self) -> None:
        """Ensure penalty system is ready for use."""
        if self.penalty_system and not hasattr(self.penalty_system, '_tables_initialized'):
            await self.penalty_system.initialize()

    async def _apply_penalty_filtering(self, candidates: List[Dict[str, Any]], 
                                     game_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply penalty system filtering to coordinate candidates."""
        try:
            if not self.penalty_system:
                # No penalty system - return all candidates
                return candidates

            # Get penalty recommendations for all candidate coordinates
            candidate_coords = [(c['x'], c['y']) for c in candidates]
            avoidance_scores = await self.penalty_system.get_avoidance_recommendations(game_id, candidate_coords)

            filtered_candidates = []
            avoided_count = 0

            for candidate in candidates:
                coord = (candidate['x'], candidate['y'])
                avoidance_score = avoidance_scores.get(coord, 0.0)
                
                # Get detailed penalty information
                penalty_info = await self.penalty_system.get_coordinate_penalty(game_id, candidate['x'], candidate['y'])
                
                # Add penalty information to candidate
                candidate['penalty_info'] = penalty_info
                candidate['avoidance_score'] = avoidance_score

                # Apply penalty-adjusted scoring
                original_score = candidate.get('total_score', 0.0)
                
                # Reduce score based on penalty (higher penalty = lower final score)
                penalty_factor = 1.0 - (avoidance_score * 0.7)  # Max 70% penalty reduction
                candidate['total_score'] = original_score * penalty_factor

                # Filter out heavily penalized coordinates (unless we need fallbacks)
                if avoidance_score < 0.8:  # Allow coordinates with penalty < 80%
                    filtered_candidates.append(candidate)
                else:
                    avoided_count += 1
                    logger.debug(f"Avoided heavily penalized coordinate ({coord[0]}, {coord[1]}) "
                               f"with penalty {penalty_info.get('penalty_score', 0):.3f}")

            if avoided_count > 0:
                self.stats['penalty_avoidances'] += avoided_count
                logger.info(f"Penalty system filtered out {avoided_count} heavily penalized coordinates")

            logger.debug(f"Penalty filtering: {len(candidates)} -> {len(filtered_candidates)} candidates")
            return filtered_candidates

        except Exception as e:
            logger.error(f"Error in penalty filtering: {e}")
            return candidates  # Return original candidates on error

    async def _get_penalty_aware_fallback_candidates(self, all_candidates: List[Dict[str, Any]], 
                                                   game_id: str) -> List[Dict[str, Any]]:
        """Get fallback candidates that consider penalty decay for recovery."""
        try:
            if not self.penalty_system:
                return all_candidates[:3]  # Just return top 3 if no penalty system

            recovery_candidates = []
            
            for candidate in all_candidates:
                penalty_info = candidate.get('penalty_info')
                if not penalty_info:
                    penalty_info = await self.penalty_system.get_coordinate_penalty(
                        game_id, candidate['x'], candidate['y']
                    )

                # Check if coordinate is eligible for recovery
                if penalty_info.get('recovery_available', False):
                    # Boost score for recovery attempts
                    candidate['total_score'] = candidate.get('total_score', 0) + 0.2
                    candidate['recovery_attempt'] = True
                    recovery_candidates.append(candidate)
                elif penalty_info.get('penalty_score', 0) < 0.5:  # Low penalty
                    recovery_candidates.append(candidate)

            # If no recovery candidates, allow some heavily penalized ones with decay
            if not recovery_candidates:
                logger.info("No recovery candidates - applying penalty decay and retrying")
                await self.penalty_system.decay_penalties(game_id)
                
                # Re-evaluate after decay
                for candidate in all_candidates[:5]:  # Top 5 candidates
                    updated_penalty = await self.penalty_system.get_coordinate_penalty(
                        game_id, candidate['x'], candidate['y']
                    )
                    if updated_penalty.get('penalty_score', 0) < 0.7:  # Allow if decay helped
                        candidate['penalty_info'] = updated_penalty
                        candidate['post_decay_attempt'] = True
                        recovery_candidates.append(candidate)

            logger.info(f"Penalty-aware fallback: {len(recovery_candidates)} recovery candidates available")
            return recovery_candidates[:3]  # Return top 3 recovery candidates

        except Exception as e:
            logger.error(f"Error in penalty-aware fallback: {e}")
            return all_candidates[:3]  # Fallback to top 3

    async def _record_penalty_system_feedback(self, coordinates: Tuple[int, int], game_id: str,
                                             effectiveness: Dict[str, Any], score_change: float,
                                             session: Dict[str, Any], change_metrics: Dict[str, Any]):
        """Record the results of coordinate attempt in penalty system for learning."""
        try:
            if not self.penalty_system:
                return  # No penalty system available

            x, y = coordinates
            
            # Determine if the action was successful
            is_successful = effectiveness.get('effective', False) and score_change >= 0
            
            # Prepare pseudo-button context data
            pseudo_button_data = {}
            
            # Check if this was a detected pseudo-button
            discovered_buttons = session.get('discovered_buttons', [])
            for button in discovered_buttons:
                if button['x'] == x and button['y'] == y:
                    pseudo_button_data = {
                        'was_pseudo_button': True,
                        'confidence': button.get('confidence', 0.0),
                        'type': button.get('type', 'unknown'),
                        'effectiveness': effectiveness.get('confidence', 0.0),
                        'context': {
                            'frame_stagnant': session.get('stagnation_count', 0) > 0,
                            'attempt_number': len(session.get('tried_pseudo_buttons', [])),
                            'change_metrics': change_metrics
                        }
                    }
                    break
            
            # Prepare general context
            context = {
                'action_type': 'ACTION6',
                'frame_stagnant': session.get('stagnation_count', 0) > 0,
                'session_attempts': len(session.get('tried_pseudo_buttons', [])),
                'effectiveness_confidence': effectiveness.get('confidence', 0.0),
                'visual_changes_detected': change_metrics.get('change_ratio', 0.0) > 0.01
            }
            
            # Record the attempt in penalty system
            penalty_result = await self.penalty_system.record_coordinate_attempt(
                game_id=game_id,
                x=x,
                y=y,
                success=is_successful,
                score_change=score_change,
                action_type='ACTION6',
                context=context,
                pseudo_button_data=pseudo_button_data
            )
            
            # Log penalty system response
            if penalty_result.get('penalty_applied', False):
                logger.info(f"Penalty system applied penalty to ({x}, {y}): "
                          f"{penalty_result.get('penalty_reason', 'unknown')} "
                          f"(score: {penalty_result.get('penalty_score', 0):.3f})")
            elif is_successful:
                logger.debug(f"Penalty system recorded successful attempt at ({x}, {y})")
                
        except Exception as e:
            logger.error(f"Error recording penalty system feedback: {e}")

    # ========== CROSS-GAME TRANSFER LEARNING METHODS ==========
    
    async def _extract_transfer_learning_patterns(self, game_id: str) -> None:
        """Extract transferable patterns from current game session."""
        try:
            session = self._get_or_create_session(game_id)
            
            # Prepare session data for pattern extraction
            session_data = {
                'pseudo_button_learning': {
                    'button_effects': await self._load_pseudo_button_learning(game_id),
                    'successful_sequences': session.get('successful_sequences', [])
                },
                'action_history': session.get('action_history', []),
                'performance_metrics': {
                    'success_rate': len(session.get('successful_sequences', [])) / max(len(session.get('tried_pseudo_buttons', [])), 1),
                    'effectiveness_score': self._calculate_session_effectiveness(session)
                }
            }
            
            # Extract patterns using transfer learning system
            patterns = self.transfer_learning.extract_patterns_from_game_session(game_id, session_data)
            
            if patterns:
                logger.info(f"Extracted {len(patterns)} transferable patterns from game {game_id}")
                
        except Exception as e:
            logger.error(f"Error extracting transfer learning patterns: {e}")

    async def _apply_transfer_learning_patterns(self, game_id: str, current_context: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Apply transferable patterns to generate coordinate candidates."""
        transfer_coordinates = []
        
        try:
            # Create game context for pattern matching
            game_context = await self._create_game_context(game_id, current_context)
            
            # Get applicable patterns
            applicable_patterns = self.transfer_learning.get_applicable_patterns(game_context)
            
            for pattern, similarity in applicable_patterns[:3]:  # Apply top 3 patterns
                try:
                    pattern_coords = self.transfer_learning.apply_pattern_to_coordinates(pattern, current_context)
                    
                    # Weight coordinates by pattern similarity and effectiveness
                    weight = similarity * pattern.effectiveness_score
                    weighted_coords = [(coord, weight) for coord in pattern_coords]
                    
                    transfer_coordinates.extend(weighted_coords)
                    
                    logger.debug(f"Applied pattern {pattern.pattern_type} with similarity {similarity:.2f}, "
                               f"generated {len(pattern_coords)} coordinates")
                               
                except Exception as e:
                    logger.error(f"Error applying pattern {pattern.pattern_id}: {e}")
            
            # Sort by weight and return top coordinates
            transfer_coordinates.sort(key=lambda x: x[1], reverse=True)
            return [coord for coord, weight in transfer_coordinates[:10]]
            
        except Exception as e:
            logger.error(f"Error applying transfer learning patterns: {e}")
            return []

    async def _create_game_context(self, game_id: str, current_context: Dict[str, Any]) -> GameContext:
        """Create GameContext for transfer learning pattern matching."""
        try:
            frame = current_context.get('current_frame', [])
            grid_size = current_context.get('grid_size', (10, 10))
            
            # Extract context features
            color_palette = set()
            object_count = 0
            
            if frame:
                for row in frame:
                    for cell in row:
                        # Normalize cell value to a scalar for hashing
                        scalar_val = None
                        if isinstance(cell, (int, float)):
                            scalar_val = int(cell)
                        elif isinstance(cell, (list, tuple)) and len(cell) > 0:
                            # pick first numeric element as representative
                            for el in cell:
                                if isinstance(el, (int, float)):
                                    scalar_val = int(el)
                                    break
                        elif isinstance(cell, str):
                            try:
                                scalar_val = int(cell)
                            except Exception:
                                scalar_val = None

                        if scalar_val and scalar_val != 0:
                            color_palette.add(scalar_val)
                            object_count += 1
            
            # Calculate complexity score
            complexity_score = min(1.0, (len(color_palette) * object_count) / 100)
            
            # Get available actions from context
            action_space = current_context.get('available_actions', ['action6'])
            
            # Extract visual features
            visual_features = {
                'dominant_colors': list(color_palette)[:5],
                'density': object_count / (grid_size[0] * grid_size[1]),
                'pattern_complexity': complexity_score
            }
            
            return GameContext(
                game_id=game_id,
                grid_size=grid_size,
                color_palette=color_palette,
                object_count=object_count,
                complexity_score=complexity_score,
                action_space=action_space,
                visual_features=visual_features
            )
            
        except Exception as e:
            logger.error(f"Error creating game context: {e}")
            # Return default context
            return GameContext(
                game_id=game_id,
                grid_size=(10, 10),
                color_palette=set([1]),
                object_count=1,
                complexity_score=0.5,
                action_space=['action6'],
                visual_features={}
            )

    async def _record_transfer_learning_feedback(self, pattern_id: str, coordinates: Tuple[int, int],
                                                success: bool, game_id: str, outcome: Dict[str, Any]) -> None:
        """Record feedback on transfer learning pattern effectiveness."""
        try:
            context = {
                'game_id': game_id,
                'coordinates': coordinates,
                'action_type': 'action6'
            }
            
            self.transfer_learning.record_transfer_feedback(
                pattern_id=pattern_id,
                success=success,
                context=context,
                outcome=outcome
            )
            
            logger.debug(f"Recorded transfer learning feedback: pattern {pattern_id}, success={success}")
            
        except Exception as e:
            logger.error(f"Error recording transfer learning feedback: {e}")

    def _calculate_session_effectiveness(self, session: Dict[str, Any]) -> float:
        """Calculate effectiveness score for current session."""
        successful_sequences = len(session.get('successful_sequences', []))
        tried_buttons = len(session.get('tried_pseudo_buttons', []))
        
        if tried_buttons == 0:
            return 0.0
            
        return successful_sequences / tried_buttons

    async def _enhance_coordinates_with_transfer_learning(self, candidates: List[Dict[str, Any]], 
                                                         game_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance coordinate candidates with transfer learning patterns."""
        try:
            # Get transfer learning coordinates
            transfer_coords = await self._apply_transfer_learning_patterns(game_id, context)
            
            # Add transfer learning coordinates to candidates
            enhanced_candidates = candidates.copy()
            
            for coord in transfer_coords:
                enhanced_candidates.append({
                    'x': coord[0],
                    'y': coord[1],
                    'confidence': 0.7,  # High confidence for transfer learning
                    'source': 'transfer_learning',
                    'reasoning': 'Generated from cross-game pattern transfer'
                })
            
            # Boost confidence of candidates that match transfer learning suggestions
            for candidate in enhanced_candidates:
                candidate_coord = (candidate['x'], candidate['y'])
                if candidate_coord in transfer_coords:
                    candidate['confidence'] = min(1.0, candidate.get('confidence', 0.5) + 0.2)
                    candidate['reasoning'] += ' + transfer learning boost'
            
            logger.debug(f"Enhanced {len(candidates)} candidates with {len(transfer_coords)} transfer learning coordinates")
            return enhanced_candidates
            
        except Exception as e:
            logger.error(f"Error enhancing coordinates with transfer learning: {e}")
            return candidates

    async def get_optimal_action6_coordinates(self, frame: List[List[int]],
                                            game_id: str,
                                            context: Dict[str, Any] = None) -> Tuple[int, int]:
        """Get optimal coordinates for Action 6 using enhanced detection.

        This is the main method for Action 6 coordinate selection that:
        1. Detects pseudo-buttons using computer vision
        2. Enhances candidates with coordinate intelligence
        3. ENHANCED: Filters candidates using penalty decay system
        4. Tests promising coordinates for effectiveness (in Action 6-only games)
        5. Returns the best coordinates based on all available information

        Args:
            frame: Current game frame
            game_id: Game identifier
            context: Additional context about the game state

        Returns:
            Tuple of (x, y) coordinates for optimal Action 6 click
        """
        try:
            self.stats['action6_selections'] += 1
            session = self._get_or_create_session(game_id)

            # ENHANCED: Ensure penalty system is ready
            await self._ensure_penalty_system_ready()

            # Check for frame stagnation - if frame isn't changing, try different pseudo-buttons
            is_stagnant = self._detect_frame_stagnation(frame, game_id)

            # NEW: Check if we should force exploration due to area stagnation
            force_exploration = self._should_force_exploration(game_id, context)

            # Check if this is an Action 6-only game
            is_action6_only = self._is_action6_only_game(context)

            # PRIORITY: If area stagnation detected, use exploration mode immediately
            if force_exploration and frame and len(frame) > 0 and len(frame[0]) > 0:
                grid_dims = (len(frame[0]), len(frame))
                try:
                    exploration_coords = self._get_strategic_action6_coordinates(grid_dims, game_id)
                    logger.info(f"🗺️ PRIORITY EXPLORATION: Breaking out of stagnation with coordinates {exploration_coords}")
                    self.stats['exploration_selections'] = self.stats.get('exploration_selections', 0) + 1
                    self.stats['forced_explorations'] = self.stats.get('forced_explorations', 0) + 1
                    return exploration_coords
                except Exception as e:
                    logger.warning(f"Priority exploration failed: {e}")
                    # Continue to pseudo-button logic as fallback

            # Step 1: Detect potential pseudo-buttons (COMPREHENSIVE - every object)
            if self.vision_detector:
                button_candidates = await self.vision_detector.detect_pseudo_buttons(frame, game_id)

                # ENHANCEMENT: Also detect EVERY object as potential pseudo-button
                all_objects = self._detect_all_objects_as_pseudo_buttons(frame, game_id)

                # Combine sophisticated detection with comprehensive object enumeration
                combined_candidates = button_candidates + all_objects
                # Remove duplicates that are too close together
                combined_candidates = self._deduplicate_all_candidates(combined_candidates)

                if combined_candidates:
                    logger.debug(f"Found {len(button_candidates)} sophisticated candidates + {len(all_objects)} all-objects = {len(combined_candidates)} total candidates")

                    # Step 2: Store all discovered buttons for this session
                    session['discovered_buttons'] = combined_candidates

                    # Step 3: Enhance candidates with coordinate intelligence
                    enhanced_candidates = self._enhance_candidates_with_intelligence(
                        combined_candidates, game_id
                    )

                    # STEP 3.5: ENHANCED - Apply penalty system filtering
                    penalty_filtered_candidates = await self._apply_penalty_filtering(
                        enhanced_candidates, game_id, context or {}
                    )

                    # STEP 3.6: ENHANCED - Apply cross-game transfer learning
                    transfer_learning_context = {
                        'current_frame': frame,
                        'grid_size': (len(frame[0]) if frame and frame[0] else 10, len(frame) if frame else 10),
                        'available_actions': context.get('available_actions', ['action6']) if context else ['action6'],
                        'reference_coordinates': [(c['x'], c['y']) for c in penalty_filtered_candidates[:5]],
                        'recent_actions': session.get('action_history', [])[-3:]
                    }

                    transfer_enhanced_candidates = await self._enhance_coordinates_with_transfer_learning(
                        penalty_filtered_candidates, game_id, transfer_learning_context
                    )

                    # Step 4: INTELLIGENT SELECTION - prioritize untried pseudo-buttons
                    untried_candidates = self._get_untried_pseudo_buttons(transfer_enhanced_candidates, game_id)

                    # ENHANCED STAGNATION HANDLING
                    if is_stagnant and untried_candidates:
                        logger.info(f"Frame stagnant - cycling to untried pseudo-button from {len(untried_candidates)} options")
                        target_candidates = untried_candidates
                    elif is_stagnant and not untried_candidates:
                        # RETRY PREVIOUSLY FAILED BUTTONS - game state may have changed!
                        logger.info(f"Frame stagnant + all buttons tried - retrying failed buttons (state may have changed)")
                        target_candidates = self._get_context_retry_candidates(transfer_enhanced_candidates, game_id)
                    elif untried_candidates:
                        logger.debug(f"Preferring untried pseudo-buttons ({len(untried_candidates)} available)")
                        target_candidates = untried_candidates
                    else:
                        # All buttons tried, re-try the most effective ones
                        logger.debug(f"All pseudo-buttons tried, selecting from most effective")
                        target_candidates = transfer_enhanced_candidates

                    # ENHANCED: If penalty filtering removed too many candidates, use penalty-aware fallback
                    if not target_candidates and transfer_enhanced_candidates != enhanced_candidates:
                        logger.info("Penalty filtering removed all candidates - using penalty-aware fallback")
                        target_candidates = await self._get_penalty_aware_fallback_candidates(
                            enhanced_candidates, game_id
                        )
                        self.stats['penalty_recoveries'] += 1

                    # Step 5: For Action 6-only games, select intelligently
                    if is_action6_only and len(target_candidates) > 0:
                        # Use the best candidate based on combined scoring (now includes penalty scores)
                        best_candidate = max(target_candidates,
                                           key=lambda c: c.get('total_score', 0))

                        coords = (best_candidate['x'], best_candidate['y'])

                        # Record this attempt for learning
                        self._record_pseudo_button_attempt(coords, game_id)

                        self.stats['button_based_selections'] += 1
                        
                        # ENHANCED: Log penalty information
                        penalty_info = best_candidate.get('penalty_info', {})
                        if penalty_info.get('penalty_score', 0) > 0:
                            logger.info(f"Selected coordinate with penalty {penalty_info.get('penalty_score', 0):.3f}: ({coords[0]}, {coords[1]})")
                        else:
                            logger.info(f"Selected penalty-free coordinate: ({coords[0]}, {coords[1]})")
                            
                        logger.info(f"Selected {'untried' if coords not in [(x, y) for x, y in session['tried_pseudo_buttons'][:-1]] else 'retried'} "
                                  f"pseudo-button: ({coords[0]}, {coords[1]}) "
                                  f"with score {best_candidate.get('total_score', 0):.3f}")

                        return coords

            # NEW: EXPLORATION MODE - Use intelligent surveying when pseudo-buttons aren't available
            if frame and len(frame) > 0 and len(frame[0]) > 0:
                grid_dims = (len(frame[0]), len(frame))

                # Try exploration coordinates first (more intelligent than random fallback)
                try:
                    exploration_coords = self._get_strategic_action6_coordinates(grid_dims, game_id)
                    logger.info(f"🗺️ EXPLORATION MODE: Using strategic coordinates {exploration_coords}")
                    self.stats['exploration_selections'] = self.stats.get('exploration_selections', 0) + 1
                    return exploration_coords
                except Exception as e:
                    logger.warning(f"Exploration mode failed: {e}")

            # Fallback: Use coordinate intelligence or random selection
            coords = await self._get_fallback_coordinates(frame, game_id, context)
            logger.debug(f"Using fallback coordinates: {coords}")
            return coords

        except Exception as e:
            logger.error(f"Error in Action 6 coordinate selection: {e}")
            return self._get_safe_fallback_coordinates(frame)

    def _get_or_create_session(self, game_id: str) -> Dict[str, Any]:
        """Get or create learning session data for a game."""
        if game_id not in self.game_sessions:
            self.game_sessions[game_id] = {
                'tried_pseudo_buttons': [],  # List of (x, y) coordinates we've tried
                'pseudo_button_effects': {},  # (x, y) -> effect description
                'successful_sequences': [],  # List of successful button sequences
                'current_sequence': [],  # Current sequence being tried
                'last_frame_hash': None,  # Hash of last frame for stagnation detection
                'stagnation_count': 0,  # How many times frame stayed same
                'last_score': 0,  # Last known score
                'discovered_buttons': [],  # All detected pseudo-buttons with their properties
                'frame_similarity_threshold': 0.95  # Threshold for considering frames "the same"
            }
        return self.game_sessions[game_id]

    def _calculate_frame_hash(self, frame: List[List[int]]) -> str:
        """Calculate a simple hash of the frame for stagnation detection."""
        try:
            import hashlib
            # Convert frame to string and hash it
            frame_str = str(frame)
            return hashlib.md5(frame_str.encode()).hexdigest()
        except:
            # Fallback: sum of all cell values
            total = 0
            for row in frame:
                for cell in row:
                    total += cell
            return str(total)

    def _detect_frame_stagnation(self, frame: List[List[int]], game_id: str) -> bool:
        """Detect if the frame has remained largely unchanged (stagnant)."""
        session = self._get_or_create_session(game_id)
        current_hash = self._calculate_frame_hash(frame)

        if session['last_frame_hash'] is None:
            session['last_frame_hash'] = current_hash
            session['stagnation_count'] = 0
            return False

        if current_hash == session['last_frame_hash']:
            session['stagnation_count'] += 1
            is_stagnant = session['stagnation_count'] >= 2  # Consider stagnant after 2 identical frames
            if is_stagnant:
                logger.info(f"Frame stagnation detected for game {game_id} (count: {session['stagnation_count']})")
            return is_stagnant
        else:
            session['last_frame_hash'] = current_hash
            session['stagnation_count'] = 0
            return False

    def _get_untried_pseudo_buttons(self, all_candidates: List[Dict[str, Any]], game_id: str) -> List[Dict[str, Any]]:
        """Get pseudo-button candidates that haven't been tried yet."""
        session = self._get_or_create_session(game_id)
        tried_coords = set((x, y) for x, y in session['tried_pseudo_buttons'])

        untried = []
        for candidate in all_candidates:
            coord = (candidate['x'], candidate['y'])
            if coord not in tried_coords:
                untried.append(candidate)

        logger.debug(f"Found {len(untried)} untried pseudo-buttons out of {len(all_candidates)} total candidates")
        return untried

    def _record_pseudo_button_attempt(self, coordinates: Tuple[int, int], game_id: str) -> None:
        """Record that we tried a specific pseudo-button."""
        session = self._get_or_create_session(game_id)
        if coordinates not in session['tried_pseudo_buttons']:
            session['tried_pseudo_buttons'].append(coordinates)
            session['current_sequence'].append(coordinates)
            logger.debug(f"Recorded attempt of pseudo-button {coordinates} for game {game_id}")

    def _record_pseudo_button_effect(self, coordinates: Tuple[int, int], game_id: str,
                                   effect_description: str, effectiveness: Dict[str, Any]) -> None:
        """Record what effect a pseudo-button had when clicked."""
        session = self._get_or_create_session(game_id)
        session['pseudo_button_effects'][coordinates] = {
            'effect': effect_description,
            'effectiveness': effectiveness,
            'attempts': session['pseudo_button_effects'].get(coordinates, {}).get('attempts', 0) + 1
        }
        logger.info(f"Recorded effect for pseudo-button {coordinates}: {effect_description}")

    def _check_sequence_success(self, score_change: float, game_id: str) -> bool:
        """Check if current sequence led to success and record it."""
        session = self._get_or_create_session(game_id)

        # Consider successful if we got positive score change
        if score_change > 0 and len(session['current_sequence']) > 0:
            # Record this as a successful sequence
            successful_sequence = session['current_sequence'].copy()
            session['successful_sequences'].append({
                'sequence': successful_sequence,
                'score_gained': score_change,
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"Recorded successful sequence for game {game_id}: {successful_sequence} (score +{score_change})")

            # Reset current sequence
            session['current_sequence'] = []
            return True
        elif score_change < 0:
            # Negative score - this sequence was bad, clear it
            logger.debug(f"Clearing unsuccessful sequence due to negative score: {session['current_sequence']}")
            session['current_sequence'] = []
            return False

        return False

    def _detect_all_objects_as_pseudo_buttons(self, frame: List[List[int]], game_id: str) -> List[Dict[str, Any]]:
        """Detect EVERY non-zero pixel/object as a potential pseudo-button."""
        try:
            if not frame or not frame[0]:
                return []

            height = len(frame)
            width = len(frame[0])
            all_objects = []

            logger.debug(f"Enumerating ALL objects in {height}x{width} frame as potential pseudo-buttons")

            # Check every pixel in the frame - every non-zero pixel is a potential pseudo-button
            for y in range(0, height, 2):  # Sample every 2 pixels for performance
                for x in range(0, width, 2):
                    if y < height and x < width:
                        cell_value = frame[y][x]
                        if isinstance(cell_value, (list, tuple)) and len(cell_value) > 0:
                            cell_value = cell_value[0] if isinstance(cell_value[0], (int, float)) else 0
                        elif not isinstance(cell_value, (int, float)):
                            cell_value = 0

                        # Every non-zero pixel is a potential pseudo-button
                        if cell_value > 0:
                            # Check if this is part of a larger object (group nearby pixels)
                            object_size = self._get_object_size(frame, x, y, cell_value)

                            all_objects.append({
                                'x': x,
                                'y': y,
                                'confidence': 0.5,  # Neutral confidence for all objects
                                'brightness': cell_value,
                                'contrast': 1,  # Default contrast
                                'type': 'comprehensive_object',
                                'object_size': object_size,
                                'color_value': cell_value
                            })

            logger.debug(f"Found {len(all_objects)} total objects as potential pseudo-buttons")
            return all_objects

        except Exception as e:
            logger.warning(f"Error in comprehensive object detection: {e}")
            return []

    def _get_object_size(self, frame: List[List[int]], start_x: int, start_y: int, target_value: int) -> int:
        """Get approximate size of object starting at coordinates."""
        try:
            height = len(frame)
            width = len(frame[0]) if frame else 0

            # Simple size estimation by checking 3x3 neighborhood
            similar_count = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = start_y + dy, start_x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        cell_value = frame[ny][nx]
                        if isinstance(cell_value, (list, tuple)) and len(cell_value) > 0:
                            cell_value = cell_value[0]
                        if abs(cell_value - target_value) <= 1:  # Similar color
                            similar_count += 1

            return similar_count

        except:
            return 1

    def _deduplicate_all_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates that are too close together."""
        if not candidates:
            return []

        unique_candidates = []
        min_distance = 4  # Smaller distance for comprehensive detection

        for candidate in candidates:
            is_duplicate = False
            for existing in unique_candidates:
                distance = ((candidate['x'] - existing['x'])**2 + (candidate['y'] - existing['y'])**2)**0.5
                if distance < min_distance:
                    # Keep the one with higher confidence or better type
                    if (candidate.get('confidence', 0) > existing.get('confidence', 0) or
                        candidate.get('type') != 'comprehensive_object'):
                        unique_candidates.remove(existing)
                        unique_candidates.append(candidate)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_candidates.append(candidate)

        return unique_candidates

    def _get_context_retry_candidates(self, all_candidates: List[Dict[str, Any]], game_id: str) -> List[Dict[str, Any]]:
        """Get pseudo-buttons to retry based on context changes (game state may have changed)."""
        session = self._get_or_create_session(game_id)

        # Get previously tried buttons that had minimal or no effect
        retry_candidates = []
        tried_coords = set((x, y) for x, y in session['tried_pseudo_buttons'])

        for candidate in all_candidates:
            coord = (candidate['x'], candidate['y'])
            if coord in tried_coords:
                # Check if this button had minimal effect before
                effect = session['pseudo_button_effects'].get(coord)
                if effect:
                    effect_desc = effect.get('effect', '')
                    # Retry buttons that had "NO_VISUAL_CHANGE" or "MINIMAL_EFFECT" - they might work now!
                    if ('NO_VISUAL_CHANGE' in effect_desc or 'MINIMAL_EFFECT' in effect_desc or
                        'low_impact' in effect_desc):
                        candidate['retry_reason'] = 'state_dependent'
                        retry_candidates.append(candidate)

        if not retry_candidates:
            # If no specific candidates to retry, retry all previously tried ones
            for candidate in all_candidates:
                coord = (candidate['x'], candidate['y'])
                if coord in tried_coords:
                    candidate['retry_reason'] = 'comprehensive_retry'
                    retry_candidates.append(candidate)

        logger.info(f"Context retry: found {len(retry_candidates)} candidates to retry (game state may have changed)")
        return retry_candidates

    def _enhance_candidates_with_intelligence(self, candidates: List[Dict[str, Any]],
                                            game_id: str) -> List[Dict[str, Any]]:
        """Enhance button candidates with coordinate intelligence data."""
        try:
            # Get coordinate intelligence from database if available
            intelligence_data = []
            if self.db_interface and hasattr(self.db_interface, 'execute_query'):
                try:
                    query = """
                    SELECT x, y, success_rate, effectiveness_score, attempts, successes
                    FROM coordinate_intelligence
                    WHERE game_id = ? OR game_id IS NULL
                    ORDER BY success_rate DESC, effectiveness_score DESC
                    LIMIT 20
                    """
                    intelligence_data = self.db_interface.execute_query(query, (game_id,))
                except Exception as e:
                    logger.debug(f"Could not get coordinate intelligence: {e}")

            for candidate in candidates:
                x, y = candidate['x'], candidate['y']

                # Find nearby intelligence data
                intelligence_score = 0.0
                for intel in intelligence_data:
                    intel_x, intel_y = intel.get('x', 0), intel.get('y', 0)
                    distance = ((x - intel_x)**2 + (y - intel_y)**2)**0.5

                    if distance < 20:  # Within 20 pixels
                        success_rate = intel.get('success_rate', 0)
                        effectiveness = intel.get('effectiveness_score', 0)
                        intelligence_score += (success_rate * effectiveness) / max(distance, 1)

                # Combine button detection confidence with intelligence
                button_confidence = candidate.get('confidence', 0.5)
                priority = candidate.get('priority', 0.5)

                # Calculate total score
                total_score = (button_confidence * 0.4 +
                             intelligence_score * 0.4 +
                             priority * 0.2)

                candidate['intelligence_score'] = intelligence_score
                candidate['total_score'] = total_score

            return candidates

        except Exception as e:
            logger.error(f"Error enhancing candidates with intelligence: {e}")
            return candidates

    async def analyze_action6_effectiveness(self, frame_before: List[List[int]],
                                          frame_after: List[List[int]],
                                          coordinates: Tuple[int, int],
                                          game_id: str,
                                          score_change: float = 0.0):
        """Analyze the effectiveness of an Action 6 click after execution.

        This method should be called after an Action 6 is executed to learn
        from the results and improve future coordinate selection.

        Args:
            frame_before: Frame before the Action 6
            frame_after: Frame after the Action 6
            coordinates: The coordinates that were clicked
            game_id: Game identifier
            score_change: Change in game score (if available)
        """
        try:
            if not frame_before or not frame_after or not self.vision_detector:
                logger.debug("Cannot analyze effectiveness without frames and detector")
                return

            x, y = coordinates
            session = self._get_or_create_session(game_id)

            # Calculate frame differences
            change_metrics = self.vision_detector.calculate_frame_differences(
                frame_before, frame_after
            )

            # Evaluate effectiveness based on frame changes
            effectiveness = self.vision_detector.evaluate_click_effectiveness(change_metrics)

            # Enhanced effectiveness analysis with pseudo-button learning
            effect_description = self._analyze_click_effect(change_metrics, score_change)

            # Factor in score change if available
            if score_change != 0:
                # Positive score change increases effectiveness
                score_factor = min(abs(score_change) / 10.0, 0.3)  # Max 0.3 bonus
                if score_change > 0:
                    effectiveness['confidence'] = min(effectiveness['confidence'] + score_factor, 1.0)
                    effectiveness['effective'] = True
                    effect_description += f" +SCORE({score_change})"
                else:
                    # Negative score change reduces effectiveness
                    effectiveness['confidence'] = max(effectiveness['confidence'] - score_factor, 0.0)
                    effect_description += f" -SCORE({score_change})"

            # ENHANCED: Record feedback in penalty system
            await self._record_penalty_system_feedback(
                coordinates, game_id, effectiveness, score_change, session, change_metrics
            )

            # Record the pseudo-button effect for learning
            self._record_pseudo_button_effect(coordinates, game_id, effect_description, effectiveness)

            # Check if this led to a successful sequence
            sequence_success = self._check_sequence_success(score_change, game_id)

            logger.info(f"ACTION 6 at {coordinates}: {effect_description} "
                       f"(effective: {effectiveness['effective']}, confidence: {effectiveness['confidence']:.3f})")

            if sequence_success:
                logger.info(f"Successful sequence completed! Score gained: {score_change}")

            # Update session's last score for next comparison
            session['last_score'] = session.get('last_score', 0) + score_change

            # Store the results in coordinate intelligence if database available
            if self.db_interface and hasattr(self.db_interface, 'execute_query'):
                try:
                    # Check if coordinate record exists
                    existing = self.db_interface.execute_query("""
                        SELECT attempts, successes, success_rate
                        FROM coordinate_intelligence
                        WHERE x = ? AND y = ? AND game_id = ?
                    """, (x, y, game_id))

                    if existing and len(existing) > 0:
                        # Update existing record
                        record = existing[0]
                        new_attempts = record['attempts'] + 1
                        new_successes = record['successes'] + (1 if effectiveness['effective'] else 0)
                        new_success_rate = new_successes / new_attempts

                        self.db_interface.execute_query("""
                            UPDATE coordinate_intelligence
                            SET attempts = ?, successes = ?, success_rate = ?,
                                effectiveness_score = ?, last_updated = CURRENT_TIMESTAMP
                            WHERE x = ? AND y = ? AND game_id = ?
                        """, (new_attempts, new_successes, new_success_rate,
                             effectiveness['confidence'], x, y, game_id))
                    else:
                        # Create new record
                        success_rate = 1.0 if effectiveness['effective'] else 0.0
                        self.db_interface.execute_query("""
                            INSERT INTO coordinate_intelligence
                            (x, y, game_id, attempts, successes, success_rate, effectiveness_score)
                            VALUES (?, ?, ?, 1, ?, ?, ?)
                        """, (x, y, game_id, 1 if effectiveness['effective'] else 0,
                             success_rate, effectiveness['confidence']))

                except Exception as e:
                    logger.debug(f"Could not update coordinate intelligence: {e}")

            # Store pseudo-button learning data in database
            await self._store_pseudo_button_learning(game_id, coordinates, effect_description, effectiveness)

            # Update statistics
            if effectiveness['effective']:
                self.stats['successful_clicks'] += 1
            else:
                self.stats['failed_clicks'] += 1

            logger.debug(f"Action 6 effectiveness analysis: ({x}, {y}) -> "
                        f"effective={effectiveness['effective']}, "
                        f"confidence={effectiveness['confidence']:.3f}, "
                        f"reason={effectiveness.get('analysis', 'unknown')}")

        except Exception as e:
            logger.error(f"Error analyzing Action 6 effectiveness: {e}")

    def _analyze_click_effect(self, change_metrics: Dict[str, float], score_change: float) -> str:
        """Analyze the effect of a click and return a description."""
        try:
            change_ratio = change_metrics.get('change_ratio', 0.0)
            avg_diff = change_metrics.get('avg_diff', 0.0)
            total_pixels_changed = change_metrics.get('significant_changes', 0)

            effects = []

            # Analyze visual changes
            if change_ratio > 0.1:
                effects.append(f"MAJOR_VISUAL_CHANGE({change_ratio:.2f})")
            elif change_ratio > 0.05:
                effects.append(f"MODERATE_VISUAL_CHANGE({change_ratio:.2f})")
            elif change_ratio > 0.01:
                effects.append(f"MINOR_VISUAL_CHANGE({change_ratio:.2f})")
            else:
                effects.append("NO_VISUAL_CHANGE")

            # Analyze intensity of changes
            if avg_diff > 3:
                effects.append("HIGH_INTENSITY")
            elif avg_diff > 1:
                effects.append("MEDIUM_INTENSITY")

            # Analyze pixel count
            if total_pixels_changed > 100:
                effects.append(f"MANY_PIXELS({total_pixels_changed})")
            elif total_pixels_changed > 20:
                effects.append(f"SOME_PIXELS({total_pixels_changed})")

            # Default if no effects detected
            if not effects:
                effects.append("MINIMAL_EFFECT")

            return " | ".join(effects)

        except Exception as e:
            return f"ANALYSIS_ERROR({e})"

    def _is_action6_only_game(self, context: Dict[str, Any]) -> bool:
        """Determine if this is an Action 6-only game."""
        if not context:
            return False

        # Check available actions - handle both integer and string formats
        available_actions = context.get('available_actions', [])
        if available_actions:
            # Convert to consistent format for comparison
            action_set = set()
            for action in available_actions:
                if isinstance(action, int):
                    action_set.add(action)
                elif isinstance(action, str):
                    if action == 'ACTION6':
                        action_set.add(6)
                    elif action.startswith('ACTION'):
                        try:
                            action_num = int(action.replace('ACTION', ''))
                            action_set.add(action_num)
                        except ValueError:
                            pass
            
            # Check if only ACTION 6 is available
            if len(action_set) == 1 and 6 in action_set:
                return True

        # Check game state
        game_state = context.get('game_state')
        if game_state and hasattr(game_state, 'available_actions'):
            actions = game_state.available_actions or []
            if len(actions) == 1:
                action = actions[0]
                if action == 6 or action == 'ACTION6':
                    return True

        # Default: assume Action 6-only if doing coordinate selection
        return True

    async def _get_fallback_coordinates(self, frame: List[List[int]],
                                      game_id: str,
                                      context: Dict[str, Any] = None) -> Tuple[int, int]:
        """Get fallback coordinates when button detection is not available."""
        # Try to get coordinates from intelligence database
        if self.db_interface and hasattr(self.db_interface, 'execute_query'):
            try:
                query = """
                SELECT x, y, success_rate
                FROM coordinate_intelligence
                WHERE game_id = ? AND success_rate > 0.5
                ORDER BY success_rate DESC, effectiveness_score DESC
                LIMIT 1
                """
                results = self.db_interface.execute_query(query, (game_id,))
                if results and len(results) > 0:
                    best = results[0]
                    return best['x'], best['y']
            except Exception as e:
                logger.debug(f"Could not get intelligence coordinates: {e}")

        # Ultimate fallback: center of frame with some randomization
        return self._get_safe_fallback_coordinates(frame)

    def _get_safe_fallback_coordinates(self, frame: List[List[int]]) -> Tuple[int, int]:
        """Get safe fallback coordinates."""
        if not frame or len(frame) == 0:
            return 25, 25  # Default safe coordinates

        height, width = len(frame), len(frame[0]) if frame else 50

        # Center with slight randomization
        import random
        center_x = width // 2
        center_y = height // 2

        # Add some randomization (±25% of frame size)
        x_offset = random.randint(-width//4, width//4)
        y_offset = random.randint(-height//4, height//4)

        x = max(5, min(width - 5, center_x + x_offset))
        y = max(5, min(height - 5, center_y + y_offset))

        return x, y

    async def _store_pseudo_button_learning(self, game_id: str, coordinates: Tuple[int, int],
                                          effect_description: str, effectiveness: Dict[str, Any]) -> None:
        """Store pseudo-button learning data in database for persistence."""
        try:
            if not self.db_interface or not hasattr(self.db_interface, 'execute_query'):
                return

            x, y = coordinates
            session = self._get_or_create_session(game_id)

            # Store pseudo-button effect data
            try:
                self.db_interface.execute_query("""
                    INSERT OR REPLACE INTO pseudo_button_learning
                    (game_id, x, y, effect_description, effectiveness_score,
                     confidence, attempts, last_used, visual_changes, score_impact)
                    VALUES (?, ?, ?, ?, ?, ?,
                           COALESCE((SELECT attempts FROM pseudo_button_learning WHERE game_id=? AND x=? AND y=?) + 1, 1),
                           CURRENT_TIMESTAMP, ?, ?)
                """, (game_id, x, y, effect_description,
                     effectiveness.get('confidence', 0), effectiveness.get('confidence', 0),
                     game_id, x, y,
                     effectiveness.get('score', 0), effectiveness.get('effective', False)))

                # Store successful sequences if any
                if len(session['successful_sequences']) > 0:
                    latest_sequence = session['successful_sequences'][-1]
                    sequence_str = str(latest_sequence['sequence'])
                    score_gained = latest_sequence['score_gained']

                    self.db_interface.execute_query("""
                        INSERT INTO pseudo_button_sequences
                        (game_id, sequence_coords, score_gained, timestamp, sequence_length)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                    """, (game_id, sequence_str, score_gained, len(latest_sequence['sequence'])))

                    logger.info(f"Stored successful sequence in database: {sequence_str} (+{score_gained} score)")

            except Exception as e:
                logger.debug(f"Could not store pseudo-button learning: {e}")

        except Exception as e:
            logger.warning(f"Error storing pseudo-button learning: {e}")

    async def _load_pseudo_button_learning(self, game_id: str) -> Dict[str, Any]:
        """Load existing pseudo-button learning data from database."""
        try:
            if not self.db_interface or not hasattr(self.db_interface, 'execute_query'):
                return {}

            # Load pseudo-button effects for this game
            button_effects = {}
            try:
                query = """
                    SELECT x, y, effect_description, effectiveness_score, confidence, attempts
                    FROM pseudo_button_learning
                    WHERE game_id = ?
                    ORDER BY effectiveness_score DESC, confidence DESC
                """
                results = self.db_interface.execute_query(query, (game_id,))

                for result in results or []:
                    x, y, effect, effectiveness, confidence, attempts = result
                    button_effects[(x, y)] = {
                        'effect': effect,
                        'effectiveness_score': effectiveness,
                        'confidence': confidence,
                        'attempts': attempts
                    }

                logger.debug(f"Loaded {len(button_effects)} pseudo-button effects for game {game_id}")

            except Exception as e:
                logger.debug(f"Could not load pseudo-button effects: {e}")

            # Load successful sequences for this game
            successful_sequences = []
            try:
                query = """
                    SELECT sequence_coords, score_gained, timestamp, sequence_length
                    FROM pseudo_button_sequences
                    WHERE game_id = ?
                    ORDER BY score_gained DESC, timestamp DESC
                    LIMIT 10
                """
                results = self.db_interface.execute_query(query, (game_id,))

                for result in results or []:
                    sequence_str, score_gained, timestamp, seq_length = result
                    try:
                        sequence_coords = eval(sequence_str)  # Convert string back to list
                        successful_sequences.append({
                            'sequence': sequence_coords,
                            'score_gained': score_gained,
                            'timestamp': timestamp,
                            'length': seq_length
                        })
                    except:
                        continue  # Skip malformed sequences

                logger.debug(f"Loaded {len(successful_sequences)} successful sequences for game {game_id}")

            except Exception as e:
                logger.debug(f"Could not load successful sequences: {e}")

            return {
                'button_effects': button_effects,
                'successful_sequences': successful_sequences
            }

        except Exception as e:
            logger.warning(f"Error loading pseudo-button learning: {e}")
            return {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about Action 6 coordinate selection."""
        total_clicks = self.stats['successful_clicks'] + self.stats['failed_clicks']

        # Calculate learning statistics across all game sessions
        total_sessions = len(self.game_sessions)
        total_tried_buttons = sum(len(session['tried_pseudo_buttons']) for session in self.game_sessions.values())
        total_successful_sequences = sum(len(session['successful_sequences']) for session in self.game_sessions.values())
        total_discovered_buttons = sum(len(session['discovered_buttons']) for session in self.game_sessions.values())

        return {
            'action6_selections': self.stats['action6_selections'],
            'button_based_selections': self.stats['button_based_selections'],
            'successful_clicks': self.stats['successful_clicks'],
            'failed_clicks': self.stats['failed_clicks'],
            'total_effectiveness_analyses': total_clicks,
            'success_rate': (
                self.stats['successful_clicks'] / max(total_clicks, 1)
            ),
            'button_usage_rate': (
                self.stats['button_based_selections'] / max(self.stats['action6_selections'], 1)
            ),
            # New learning statistics
            'learning_sessions': total_sessions,
            'total_pseudo_buttons_tried': total_tried_buttons,
            'total_successful_sequences': total_successful_sequences,
            'total_buttons_discovered': total_discovered_buttons,
            'avg_buttons_per_session': total_tried_buttons / max(total_sessions, 1),
            'sequence_success_rate': total_successful_sequences / max(total_sessions, 1),
            # NEW: Exploration statistics
            'exploration_selections': self.stats.get('exploration_selections', 0),
            'forced_explorations': self.stats.get('forced_explorations', 0),
            'exploration_usage_rate': (
                self.stats.get('exploration_selections', 0) / max(self.stats['action6_selections'], 1)
            ),
            'forced_exploration_rate': (
                self.stats.get('forced_explorations', 0) / max(self.stats['action6_selections'], 1)
            ),
            'total_boundaries_mapped': sum(
                len(getattr(self, 'boundary_system', {}).get('boundary_data', {}).get(game_id, {}))
                for game_id in self.game_sessions.keys()
            ) if hasattr(self, 'boundary_system') else 0
        }

    # =========================================================================
    # EXPLORATION AND MAPPING FEATURES
    # =========================================================================

    def _ensure_boundary_system_initialized(self, game_id: str) -> Dict[str, Any]:
        """Ensure boundary detection system is initialized for exploration."""
        if not hasattr(self, 'boundary_system'):
            self.boundary_system = {
                'boundary_data': {},
                'coordinate_attempts': {},
                'action_coordinate_history': {},
                'stuck_patterns': {},
                'success_zone_mapping': {},
                'last_coordinates': {},
                'safe_regions': {},
                'directional_systems': {
                    6: {
                        'current_direction': {},
                        'direction_progression': {
                            'right': {'next': 'down', 'coordinate_delta': (1, 0)},
                            'down': {'next': 'left', 'coordinate_delta': (0, 1)},
                            'left': {'next': 'up', 'coordinate_delta': (-1, 0)},
                            'up': {'next': 'right', 'coordinate_delta': (0, -1)}
                        }
                    }
                }
            }

        # Initialize game-specific data
        if game_id not in self.boundary_system['boundary_data']:
            self.boundary_system['boundary_data'][game_id] = {}
            self.boundary_system['coordinate_attempts'][game_id] = {}
            self.boundary_system['action_coordinate_history'][game_id] = {}
            self.boundary_system['stuck_patterns'][game_id] = {}
            self.boundary_system['success_zone_mapping'][game_id] = {}
            self.boundary_system['last_coordinates'][game_id] = None
            self.boundary_system['safe_regions'][game_id] = {}

        # Initialize Action 6 directional system for this game
        if game_id not in self.boundary_system['directional_systems'][6]['current_direction']:
            self.boundary_system['directional_systems'][6]['current_direction'][game_id] = 'right'

        return self.boundary_system

    def _get_strategic_action6_coordinates(self, grid_dims: Tuple[int, int], game_id: str) -> Tuple[int, int]:
        """
        INTELLIGENT SURVEYING SYSTEM for ACTION 6 - Fast grid exploration instead of slow directional crawling.

        Key Features:
        1. Jumps intelligently across the grid to map regions quickly
        2. Uses safe regions to launch exploration into unknown territory
        3. Avoids slow line-by-line traversal through known safe areas
        4. Prioritizes boundary detection and territory expansion
        """
        grid_width, grid_height = grid_dims

        # Ensure boundary system is initialized
        boundary_system = self._ensure_boundary_system_initialized(game_id)

        known_boundaries = set(boundary_system['boundary_data'][game_id].keys())
        safe_regions = boundary_system['safe_regions'][game_id]

        # INTELLIGENT SURVEYING: Instead of slow directional movement, make strategic jumps
        if safe_regions and len(list(safe_regions.values())[0]['coordinates']) > 10:
            # We have established safe regions - time to survey efficiently!
            survey_target = self._get_intelligent_survey_target(game_id, grid_dims, known_boundaries, safe_regions)

            if survey_target:
                survey_x, survey_y, survey_reason = survey_target
                logger.info(f"🗺️ ACTION 6 INTELLIGENT SURVEY: {survey_reason}")

                # Update position tracking for future moves
                self._current_game_x = survey_x
                self._current_game_y = survey_y

                return (survey_x, survey_y)

        # FALLBACK: If no safe regions yet, use improved initial exploration
        # Get current position from game state (stored by previous ACTION 6 or start at center)
        current_x = getattr(self, '_current_game_x', grid_width // 2)
        current_y = getattr(self, '_current_game_y', 0)  # Start at top for systematic mapping

        # Use directional system for ACTION 6
        directional_system = boundary_system['directional_systems'][6]
        current_direction = directional_system['current_direction'][game_id]
        direction_progression = directional_system['direction_progression']

        direction_info = direction_progression[current_direction]
        dx, dy = direction_info['coordinate_delta']

        # Calculate next coordinate in current direction
        new_x = current_x + dx
        new_y = current_y + dy

        # Check grid bounds and adjust if we hit the edge
        hit_boundary = False
        if new_x < 0 or new_x >= grid_width or new_y < 0 or new_y >= grid_height:
            hit_boundary = True
            boundary_type = f"grid_edge_{current_direction}"

            # Clamp to grid bounds
            new_x = max(0, min(new_x, grid_width - 1))
            new_y = max(0, min(new_y, grid_height - 1))

            # Mark this as a boundary
            boundary_coord = (new_x, new_y)
            boundary_system['boundary_data'][game_id][boundary_coord] = {
                'boundary_type': boundary_type,
                'detection_count': boundary_system['boundary_data'][game_id].get(boundary_coord, {}).get('detection_count', 0) + 1,
                'timestamp': time.time(),
                'action': 6
            }

            logger.info(f"🚧 ACTION 6 BOUNDARY: Hit {boundary_type} at ({new_x},{new_y}) - pivoting direction")

        # PIVOT TO NEW DIRECTION if boundary hit
        if hit_boundary:
            # Pivot to next semantic direction
            next_direction = direction_info['next']
            directional_system['current_direction'][game_id] = next_direction

            # Calculate coordinates in new direction from current position
            next_direction_info = direction_progression[next_direction]
            dx, dy = next_direction_info['coordinate_delta']

            pivot_x = current_x + dx
            pivot_y = current_y + dy

            # Ensure pivot coordinates are within bounds
            pivot_x = max(0, min(pivot_x, grid_width - 1))
            pivot_y = max(0, min(pivot_y, grid_height - 1))

            new_x, new_y = pivot_x, pivot_y
            logger.info(f"🔄 ACTION 6 PIVOT: Direction {current_direction} → {next_direction}, coordinates ({current_x},{current_y}) → ({new_x},{new_y})")

        # Update tracking data
        boundary_system['last_coordinates'][game_id] = (new_x, new_y)

        # Track coordinate attempt history
        coord_key = (new_x, new_y)
        if coord_key not in boundary_system['coordinate_attempts'][game_id]:
            boundary_system['coordinate_attempts'][game_id][coord_key] = {'attempts': 0, 'consecutive_stuck': 0}
        boundary_system['coordinate_attempts'][game_id][coord_key]['attempts'] += 1

        # Update current position tracking
        self._current_game_x = new_x
        self._current_game_y = new_y

        # CRITICAL: Detect coordinate stagnation and force movement
        if coord_key in boundary_system['coordinate_attempts'][game_id]:
            consecutive_stuck = boundary_system['coordinate_attempts'][game_id][coord_key].get('consecutive_stuck', 0)
            if consecutive_stuck > 10:  # Stuck at same coordinates for 10+ attempts
                logger.warning(f"⚠️ COORDINATE STAGNATION DETECTED at ({new_x},{new_y}) - FORCING MOVEMENT")

                # Force jump to a completely different region
                jump_regions = [
                    (grid_width // 8, grid_height // 8),      # Far corner
                    (7 * grid_width // 8, grid_height // 8),  # Opposite corner
                    (grid_width // 2, grid_height // 8),      # Top center
                    (grid_width // 8, grid_height // 2),      # Left center
                    (7 * grid_width // 8, 7 * grid_height // 8), # Far bottom right
                ]
                new_x, new_y = random.choice(jump_regions)

                # Reset stagnation counter
                boundary_system['coordinate_attempts'][game_id][coord_key]['consecutive_stuck'] = 0

                # Reset direction to explore from new position
                directional_system['current_direction'][game_id] = random.choice(['right', 'down', 'left', 'up'])

                logger.info(f"🚀 EMERGENCY JUMP: Moved to ({new_x},{new_y}), new direction: {directional_system['current_direction'][game_id]}")

                # Update tracking for new position
                self._current_game_x = new_x
                self._current_game_y = new_y
            else:
                boundary_system['coordinate_attempts'][game_id][coord_key]['consecutive_stuck'] = consecutive_stuck + 1

        # Display boundary intelligence
        num_boundaries = len(boundary_system['boundary_data'][game_id])
        direction_display = current_direction.upper()
        if hit_boundary:
            direction_display = f"{current_direction.upper()}→{boundary_system['directional_systems'][6]['current_direction'][game_id].upper()}"

        logger.info(f"🧭 ACTION 6 BOUNDARY-AWARE: {direction_display} from ({current_x},{current_y}) → ({new_x},{new_y}) | Boundaries mapped: {num_boundaries}")

        return (new_x, new_y)

    def _get_intelligent_survey_target(self, game_id: str, grid_dims: Tuple[int, int],
                                     known_boundaries: set, safe_regions: dict) -> Optional[Tuple[int, int, str]]:
        """
        Intelligent surveying: Instead of slow traversal through safe zones, jump to explore boundaries.

        Strategy:
        1. Find edges of safe regions and jump outward to test new territory
        2. Make large coordinate jumps to quickly map the grid
        3. Target unexplored quadrants
        4. Push the bounds of understanding by testing boundary extensions

        Returns: (x, y, reason) or None if no good survey target
        """
        grid_width, grid_height = grid_dims

        # Strategy 1: Jump from safe region edges to unexplored territory
        for region_id, region_data in safe_regions.items():
            region_coords = region_data['coordinates']

            # Find the extremes of this safe region
            min_x = min(coord[0] for coord in region_coords)
            max_x = max(coord[0] for coord in region_coords)
            min_y = min(coord[1] for coord in region_coords)
            max_y = max(coord[1] for coord in region_coords)

            # Create jump targets that extend beyond the safe region
            jump_targets = [
                (min_x - 5, min_y - 3, f"Jump LEFT from safe region {region_id}"),
                (max_x + 5, min_y - 3, f"Jump RIGHT from safe region {region_id}"),
                (min_x - 3, min_y - 5, f"Jump UP from safe region {region_id}"),
                (min_x - 3, max_y + 5, f"Jump DOWN from safe region {region_id}"),
                # Diagonal jumps for comprehensive mapping
                (min_x - 4, min_y - 4, f"Jump UP-LEFT from safe region {region_id}"),
                (max_x + 4, min_y - 4, f"Jump UP-RIGHT from safe region {region_id}"),
                (min_x - 4, max_y + 4, f"Jump DOWN-LEFT from safe region {region_id}"),
                (max_x + 4, max_y + 4, f"Jump DOWN-RIGHT from safe region {region_id}")
            ]

            # Find valid jump targets that are in bounds and not near known boundaries
            for target_x, target_y, reason in jump_targets:
                # Clamp to grid bounds
                target_x = max(0, min(target_x, grid_width - 1))
                target_y = max(0, min(target_y, grid_height - 1))

                # Check if this is far enough from known boundaries
                if self._is_good_survey_target((target_x, target_y), known_boundaries, min_distance=3):
                    return (target_x, target_y, reason)

        # Strategy 2: Quadrant exploration - jump to unexplored grid quadrants
        quadrants = [
            (grid_width // 4, grid_height // 4, "Explore TOP-LEFT quadrant"),
            (3 * grid_width // 4, grid_height // 4, "Explore TOP-RIGHT quadrant"),
            (grid_width // 4, 3 * grid_height // 4, "Explore BOTTOM-LEFT quadrant"),
            (3 * grid_width // 4, 3 * grid_height // 4, "Explore BOTTOM-RIGHT quadrant")
        ]

        for quad_x, quad_y, reason in quadrants:
            if self._is_good_survey_target((quad_x, quad_y), known_boundaries, min_distance=5):
                # Check if this quadrant is underexplored
                quadrant_explored = any(
                    abs(coord[0] - quad_x) < 8 and abs(coord[1] - quad_y) < 8
                    for safe_coords in [region_data['coordinates'] for region_data in safe_regions.values()]
                    for coord in safe_coords
                )

                if not quadrant_explored:
                    return (quad_x, quad_y, reason)

        # Strategy 3: Boundary extension - test just beyond known boundaries to find limits
        boundary_extensions = []
        for boundary_coord in known_boundaries:
            bx, by = boundary_coord

            # Try coordinates just beyond the boundary in multiple directions
            extensions = [
                (bx - 2, by, f"Test boundary extension LEFT of {boundary_coord}"),
                (bx + 2, by, f"Test boundary extension RIGHT of {boundary_coord}"),
                (bx, by - 2, f"Test boundary extension UP of {boundary_coord}"),
                (bx, by + 2, f"Test boundary extension DOWN of {boundary_coord}")
            ]

            for ext_x, ext_y, reason in extensions:
                # Clamp to grid bounds
                ext_x = max(0, min(ext_x, grid_width - 1))
                ext_y = max(0, min(ext_y, grid_height - 1))

                if self._is_good_survey_target((ext_x, ext_y), known_boundaries, min_distance=2):
                    boundary_extensions.append((ext_x, ext_y, reason))

        if boundary_extensions:
            return random.choice(boundary_extensions)

        return None

    def _is_good_survey_target(self, target_coord: Tuple[int, int], known_boundaries: set, min_distance: int = 3) -> bool:
        """Check if a coordinate is a good survey target (not too close to known boundaries)."""
        target_x, target_y = target_coord

        for boundary_coord in known_boundaries:
            bx, by = boundary_coord
            distance = abs(target_x - bx) + abs(target_y - by)
            if distance < min_distance:
                return False

        return True

    def _detect_area_stagnation(self, game_id: str, recent_coords_limit: int = 8, area_threshold: int = 12) -> bool:
        """
        Detect if the system is stuck in a small area (moving back and forth in same region).

        This is different from frame stagnation - we can have different frames but still be
        stuck in the same small coordinate area.

        Args:
            game_id: Game identifier
            recent_coords_limit: How many recent coordinates to analyze
            area_threshold: Maximum area size to consider "stuck"

        Returns:
            True if stuck in small area, False otherwise
        """
        session = self._get_or_create_session(game_id)

        # Get recent coordinate attempts
        recent_coords = session.get('tried_pseudo_buttons', [])[-recent_coords_limit:]

        if len(recent_coords) < 5:  # Need at least 5 coordinates to detect pattern
            return False

        # Calculate the bounding box of recent coordinates
        if recent_coords:
            x_coords = [coord[0] for coord in recent_coords]
            y_coords = [coord[1] for coord in recent_coords]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            area_width = max_x - min_x + 1
            area_height = max_y - min_y + 1
            area_size = area_width * area_height

            # Check if we're confined to a small area
            if area_size <= area_threshold:
                logger.warning(f"🔒 AREA STAGNATION DETECTED: Confined to {area_width}x{area_height} area (size={area_size}) over {len(recent_coords)} moves")
                logger.warning(f"🔒 Area bounds: X:{min_x}-{max_x}, Y:{min_y}-{max_y}")
                return True

        return False

    def _should_force_exploration(self, game_id: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if we should force exploration mode even when pseudo-buttons are available.

        This helps break out of repetitive patterns and small-area confinement.
        """
        session = self._get_or_create_session(game_id)

        # Check multiple stagnation indicators
        area_stuck = self._detect_area_stagnation(game_id)
        frame_stuck = session.get('stagnation_count', 0) > 2

        # Check if we've been trying the same small set of coordinates repeatedly
        recent_coords = session.get('tried_pseudo_buttons', [])[-12:]
        unique_recent = set(recent_coords[-8:]) if len(recent_coords) >= 8 else set(recent_coords)
        coordinate_repetition = len(recent_coords) >= 5 and len(unique_recent) <= 5

        # DEBUG: Log stagnation check details
        if len(recent_coords) >= 3:  # Only log if we have some history
            logger.debug(f"🔍 Stagnation check for {game_id}: area_stuck={area_stuck}, frame_stuck={frame_stuck}, coord_rep={coordinate_repetition}")
            logger.debug(f"🔍 Recent coords: {len(recent_coords)} total, {len(unique_recent)} unique in last 8")
            if recent_coords:
                logger.debug(f"🔍 Last few coordinates: {recent_coords[-5:]}")

        if area_stuck:
            logger.info(f"🗺️ FORCING EXPLORATION: Area stagnation detected")
            return True
        elif coordinate_repetition:
            logger.info(f"🗺️ FORCING EXPLORATION: Coordinate repetition - only {len(unique_recent)} unique coords in last {len(recent_coords)} moves")
            return True
        elif frame_stuck:
            logger.info(f"🗺️ FORCING EXPLORATION: Frame stagnation count = {session.get('stagnation_count', 0)}")
            return True

        return False


def create_action6_coordinator(db_interface=None, vision_detector=None) -> Action6Coordinator:
    """Factory function to create an Action 6 coordinator.

    Args:
        db_interface: Optional database interface for learning integration
        vision_detector: Optional pseudo-button detector instance

    Returns:
        Configured Action6Coordinator instance
    """
    return Action6Coordinator(db_interface=db_interface, vision_detector=vision_detector)