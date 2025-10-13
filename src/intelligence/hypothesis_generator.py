"""
Hypothesis Generator for Game-Specific Strategy Testing

This module automatically generates testable hypotheses about winning strategies for ARC games
based on visual pattern analysis, game mechanics detection, and historical game data.
Integrates with database to retrieve past successful hypotheses and strategies.

Key Features:
- Automatic hypothesis generation based on game mechanics
- Database integration for retrieving successful past strategies
- Hypothesis scoring and prioritization
- Integration with GamePatternAnalyzer and existing GameTypeClassifier
- Support for different hypothesis types (coordinate-based, pattern-based, sequence-based)
"""

import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from datetime import datetime, timedelta
import numpy as np

# Disable pycache
import sys
sys.dont_write_bytecode = True

# Import our new pattern analyzer
from .game_pattern_analyzer import GamePatternAnalyzer, GameMechanic, GameMechanicsProfile, get_game_pattern_analyzer

# Import existing game type classifier
from ..learning.game_type_classifier import GameTypeClassifier, get_game_type_classifier

logger = logging.getLogger(__name__)

class HypothesisType(Enum):
    """Types of hypotheses that can be generated."""
    COORDINATE_SEQUENCE = "coordinate_sequence"
    PATTERN_COMPLETION = "pattern_completion"
    OBJECT_MANIPULATION = "object_manipulation"
    COLOR_MATCHING = "color_matching"
    SHAPE_TRANSFORMATION = "shape_transformation"
    PHYSICS_SIMULATION = "physics_simulation"
    SPATIAL_REASONING = "spatial_reasoning"
    NAVIGATION_PATH = "navigation_path"

class HypothesisSource(Enum):
    """Source of hypothesis generation."""
    PATTERN_ANALYSIS = "pattern_analysis"
    DATABASE_RETRIEVAL = "database_retrieval"
    GAME_TYPE_KNOWLEDGE = "game_type_knowledge"
    HYBRID_APPROACH = "hybrid_approach"

@dataclass
class Hypothesis:
    """Represents a testable hypothesis about game strategy."""
    hypothesis_id: str
    hypothesis_type: HypothesisType
    source: HypothesisSource
    description: str
    predicted_coordinates: List[Tuple[int, int]]
    expected_actions: List[str]
    confidence: float
    reasoning: str
    supporting_evidence: Dict[str, Any]
    game_mechanics: List[GameMechanic]
    created_at: datetime
    tested: bool = False
    success_rate: float = 0.0
    test_count: int = 0

@dataclass
class HypothesisContext:
    """Context information for hypothesis generation."""
    game_id: str
    game_type: str
    mechanics_profile: GameMechanicsProfile
    screenshot_features: Dict[str, Any]
    historical_successes: List[Dict[str, Any]]
    similar_games_data: List[Dict[str, Any]]

class HypothesisGenerator:
    """
    Generates testable hypotheses about game winning strategies.

    Combines visual pattern analysis, game type classification, and database
    knowledge to automatically generate strategic hypotheses.
    """

    def __init__(self, db_connection: Optional[sqlite3.Connection] = None):
        self.db_connection = db_connection
        self.pattern_analyzer = get_game_pattern_analyzer()
        self.game_type_classifier = get_game_type_classifier()
        self.hypothesis_cache: Dict[str, List[Hypothesis]] = {}
        self.generation_strategies = self._initialize_generation_strategies()

        logger.info("Hypothesis Generator initialized")

    def _initialize_generation_strategies(self) -> Dict[GameMechanic, Dict[str, Any]]:
        """Initialize hypothesis generation strategies for different game mechanics."""
        return {
            GameMechanic.PATTERN_COMPLETION: {
                "priority_strategies": ["complete_missing_elements", "extend_sequences", "fill_symmetrical_gaps"],
                "coordinate_patterns": ["grid_completion", "sequence_continuation"],
                "evidence_weights": {"symmetry": 0.3, "repetition": 0.4, "grid_structure": 0.3}
            },
            GameMechanic.PHYSICS_SIMULATION: {
                "priority_strategies": ["simulate_gravity", "predict_collisions", "trace_movement_paths"],
                "coordinate_patterns": ["falling_trajectory", "collision_points"],
                "evidence_weights": {"object_density": 0.4, "spatial_relationships": 0.3, "movement_indicators": 0.3}
            },
            GameMechanic.SPATIAL_PUZZLE: {
                "priority_strategies": ["fit_pieces_together", "rotate_objects", "spatial_alignment"],
                "coordinate_patterns": ["geometric_fitting", "rotation_centers"],
                "evidence_weights": {"shape_complexity": 0.4, "geometric_patterns": 0.3, "spatial_gaps": 0.3}
            },
            GameMechanic.OBJECT_MANIPULATION: {
                "priority_strategies": ["move_discrete_objects", "transform_shapes", "reposition_elements"],
                "coordinate_patterns": ["object_centers", "movement_destinations"],
                "evidence_weights": {"object_boundaries": 0.4, "movement_space": 0.3, "object_interactions": 0.3}
            },
            GameMechanic.SORTING_ORGANIZING: {
                "priority_strategies": ["group_by_properties", "sort_sequences", "organize_patterns"],
                "coordinate_patterns": ["grouping_centers", "sorting_lines"],
                "evidence_weights": {"color_distribution": 0.3, "size_patterns": 0.3, "position_logic": 0.4}
            },
            GameMechanic.NAVIGATION_PATHFINDING: {
                "priority_strategies": ["find_optimal_path", "avoid_obstacles", "connect_endpoints"],
                "coordinate_patterns": ["path_waypoints", "obstacle_boundaries"],
                "evidence_weights": {"path_clarity": 0.4, "obstacle_density": 0.3, "endpoint_visibility": 0.3}
            }
        }

    def generate_hypotheses(self, game_id: str, screenshot_array: np.ndarray,
                          max_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Generate hypotheses for a given game.

        Args:
            game_id: Unique identifier for the game
            screenshot_array: Game screenshot as numpy array
            max_hypotheses: Maximum number of hypotheses to generate

        Returns:
            List of generated hypotheses, sorted by confidence
        """
        logger.info(f"Generating hypotheses for game {game_id}")

        # Check cache first
        if game_id in self.hypothesis_cache:
            logger.debug(f"Using cached hypotheses for game {game_id}")
            return self.hypothesis_cache[game_id][:max_hypotheses]

        # Analyze game mechanics and patterns
        mechanics_profile = self.pattern_analyzer.analyze_game_screenshot(screenshot_array, game_id)

        # Get game type classification
        game_type = self.game_type_classifier.extract_game_type(game_id)

        # Retrieve historical data
        historical_successes = self._get_historical_successes(game_type)
        similar_games_data = self._get_similar_games_data(game_type)

        # Create context for hypothesis generation
        context = HypothesisContext(
            game_id=game_id,
            game_type=game_type,
            mechanics_profile=mechanics_profile,
            screenshot_features=mechanics_profile.grid_features,
            historical_successes=historical_successes,
            similar_games_data=similar_games_data
        )

        # Generate hypotheses from different sources
        hypotheses = []

        # 1. Pattern-based hypotheses
        pattern_hypotheses = self._generate_pattern_based_hypotheses(context)
        hypotheses.extend(pattern_hypotheses)

        # 2. Database-retrieved hypotheses
        db_hypotheses = self._generate_database_hypotheses(context)
        hypotheses.extend(db_hypotheses)

        # 3. Game type knowledge hypotheses
        knowledge_hypotheses = self._generate_knowledge_based_hypotheses(context)
        hypotheses.extend(knowledge_hypotheses)

        # 4. Hybrid hypotheses (combining multiple sources)
        hybrid_hypotheses = self._generate_hybrid_hypotheses(context, hypotheses)
        hypotheses.extend(hybrid_hypotheses)

        # Score and rank hypotheses
        scored_hypotheses = self._score_and_rank_hypotheses(hypotheses, context)

        # Cache results
        self.hypothesis_cache[game_id] = scored_hypotheses

        # Return top hypotheses
        result = scored_hypotheses[:max_hypotheses]
        logger.info(f"Generated {len(result)} hypotheses for game {game_id}")

        return result

    def _generate_pattern_based_hypotheses(self, context: HypothesisContext) -> List[Hypothesis]:
        """Generate hypotheses based on visual pattern analysis."""
        hypotheses = []
        mechanics_profile = context.mechanics_profile
        primary_mechanic = mechanics_profile.primary_mechanic

        if primary_mechanic in self.generation_strategies:
            strategy = self.generation_strategies[primary_mechanic]

            for i, strategy_name in enumerate(strategy["priority_strategies"]):
                hypothesis = self._create_pattern_hypothesis(
                    context, primary_mechanic, strategy_name, i
                )
                if hypothesis:
                    hypotheses.append(hypothesis)

        # Generate hypotheses for secondary mechanics
        for secondary_mechanic in mechanics_profile.secondary_mechanics:
            if secondary_mechanic in self.generation_strategies:
                strategy = self.generation_strategies[secondary_mechanic]
                hypothesis = self._create_pattern_hypothesis(
                    context, secondary_mechanic, strategy["priority_strategies"][0],
                    priority_offset=10
                )
                if hypothesis:
                    hypotheses.append(hypothesis)

        return hypotheses

    def _create_pattern_hypothesis(self, context: HypothesisContext,
                                 mechanic: GameMechanic, strategy_name: str,
                                 priority_offset: int = 0) -> Optional[Hypothesis]:
        """Create a hypothesis based on a specific mechanic and strategy."""

        # Generate predicted coordinates based on mechanic type
        coordinates = self._predict_coordinates_for_mechanic(
            context, mechanic, strategy_name
        )

        if not coordinates:
            return None

        # Generate expected actions
        actions = self._predict_actions_for_strategy(mechanic, strategy_name)

        # Calculate confidence based on pattern strength
        confidence = self._calculate_pattern_confidence(context, mechanic, strategy_name)

        # Generate reasoning
        reasoning = self._generate_reasoning(context, mechanic, strategy_name, coordinates)

        # Collect supporting evidence
        evidence = self._collect_supporting_evidence(context, mechanic, strategy_name)

        hypothesis_id = f"{context.game_id}_{mechanic.value}_{strategy_name}_{priority_offset}"

        return Hypothesis(
            hypothesis_id=hypothesis_id,
            hypothesis_type=self._map_mechanic_to_hypothesis_type(mechanic),
            source=HypothesisSource.PATTERN_ANALYSIS,
            description=f"Apply {strategy_name} strategy for {mechanic.value}",
            predicted_coordinates=coordinates,
            expected_actions=actions,
            confidence=confidence,
            reasoning=reasoning,
            supporting_evidence=evidence,
            game_mechanics=[mechanic],
            created_at=datetime.now()
        )

    def _predict_coordinates_for_mechanic(self, context: HypothesisContext,
                                        mechanic: GameMechanic, strategy: str) -> List[Tuple[int, int]]:
        """Predict coordinates based on mechanic type and strategy."""
        coordinates = []
        visual_patterns = context.mechanics_profile.visual_patterns

        if mechanic == GameMechanic.PATTERN_COMPLETION:
            # Look for gaps in patterns or incomplete grids
            for pattern in visual_patterns:
                if pattern.pattern_type in ["rectangle", "row_sequence", "col_sequence"]:
                    # Predict coordinates to complete the pattern
                    x, y = pattern.location
                    w, h = pattern.size

                    if strategy == "complete_missing_elements":
                        # Add coordinates around the pattern
                        coordinates.extend([
                            (x + w, y), (x, y + h), (x + w//2, y + h//2)
                        ])
                    elif strategy == "extend_sequences":
                        # Extend the pattern
                        coordinates.extend([
                            (x + w + 5, y), (x, y + h + 5)
                        ])

        elif mechanic == GameMechanic.OBJECT_MANIPULATION:
            # Find object centers and predict movement destinations
            for pattern in visual_patterns:
                if pattern.pattern_type == "color_region":
                    x, y = pattern.location
                    w, h = pattern.size
                    center = (x + w//2, y + h//2)

                    if strategy == "move_discrete_objects":
                        # Predict nearby movement destinations
                        coordinates.extend([
                            (center[0] + 20, center[1]),
                            (center[0], center[1] + 20),
                            (center[0] - 20, center[1])
                        ])

        elif mechanic == GameMechanic.PHYSICS_SIMULATION:
            # Predict gravity effects and collision points
            for pattern in visual_patterns:
                if pattern.pattern_type == "circle":
                    x, y = pattern.location

                    if strategy == "simulate_gravity":
                        # Predict falling trajectory
                        coordinates.extend([
                            (x, y + 10), (x, y + 20), (x, y + 30)
                        ])

        elif mechanic == GameMechanic.SPATIAL_PUZZLE:
            # Find geometric centers and fitting positions
            geometric_patterns = [p for p in visual_patterns if p.pattern_type in ["rectangle", "circle"]]

            if len(geometric_patterns) >= 2:
                for i, pattern in enumerate(geometric_patterns[:3]):
                    x, y = pattern.location
                    w, h = pattern.size

                    if strategy == "fit_pieces_together":
                        # Predict fitting positions
                        coordinates.extend([
                            (x + w, y), (x, y + h)
                        ])

        elif mechanic == GameMechanic.SORTING_ORGANIZING:
            # Find grouping centers and sorting lines
            color_regions = [p for p in visual_patterns if p.pattern_type == "color_region"]

            if len(color_regions) > 1:
                # Calculate potential grouping centers
                x_coords = [p.location[0] for p in color_regions]
                y_coords = [p.location[1] for p in color_regions]

                center_x = sum(x_coords) // len(x_coords)
                center_y = sum(y_coords) // len(y_coords)

                coordinates.extend([
                    (center_x, center_y),
                    (min(x_coords), center_y),
                    (max(x_coords), center_y)
                ])

        elif mechanic == GameMechanic.NAVIGATION_PATHFINDING:
            # Find path endpoints and waypoints
            if context.screenshot_features.get("grid_detected", False):
                # Simple pathfinding coordinates
                grid_size = context.screenshot_features.get("grid_size", (10, 10))

                if grid_size[0] > 0 and grid_size[1] > 0:
                    coordinates.extend([
                        (0, 0),  # Start corner
                        (grid_size[1]-1, grid_size[0]-1),  # End corner
                        (grid_size[1]//2, grid_size[0]//2),  # Center waypoint
                    ])

        # Limit coordinates and ensure they're valid
        coordinates = coordinates[:10]  # Limit to top 10
        valid_coordinates = [(x, y) for x, y in coordinates if x >= 0 and y >= 0]

        return valid_coordinates

    def _predict_actions_for_strategy(self, mechanic: GameMechanic, strategy: str) -> List[str]:
        """Predict action sequence for a given strategy."""
        action_map = {
            GameMechanic.PATTERN_COMPLETION: {
                "complete_missing_elements": ["click", "drag", "release"],
                "extend_sequences": ["click", "click", "click"],
                "fill_symmetrical_gaps": ["click", "mirror_action", "click"]
            },
            GameMechanic.OBJECT_MANIPULATION: {
                "move_discrete_objects": ["click_object", "drag_to_position", "release"],
                "transform_shapes": ["click_object", "rotate", "confirm"],
                "reposition_elements": ["select_object", "move", "place"]
            },
            GameMechanic.PHYSICS_SIMULATION: {
                "simulate_gravity": ["release_object", "wait", "observe"],
                "predict_collisions": ["calculate_trajectory", "position_object", "release"],
                "trace_movement_paths": ["follow_path", "predict_endpoint", "interact"]
            },
            GameMechanic.SPATIAL_PUZZLE: {
                "fit_pieces_together": ["select_piece", "rotate_if_needed", "position", "confirm"],
                "rotate_objects": ["select_object", "rotate", "check_fit", "confirm"],
                "spatial_alignment": ["align_x", "align_y", "confirm_position"]
            },
            GameMechanic.SORTING_ORGANIZING: {
                "group_by_properties": ["identify_groups", "move_to_group", "confirm"],
                "sort_sequences": ["identify_order", "rearrange", "confirm"],
                "organize_patterns": ["recognize_pattern", "apply_organization", "verify"]
            },
            GameMechanic.NAVIGATION_PATHFINDING: {
                "find_optimal_path": ["identify_start", "identify_end", "trace_path", "execute"],
                "avoid_obstacles": ["detect_obstacles", "plan_around", "execute_path"],
                "connect_endpoints": ["find_endpoints", "plan_connection", "execute"]
            }
        }

        return action_map.get(mechanic, {}).get(strategy, ["click", "observe", "analyze"])

    def _calculate_pattern_confidence(self, context: HypothesisContext,
                                    mechanic: GameMechanic, strategy: str) -> float:
        """Calculate confidence score for a pattern-based hypothesis."""
        base_confidence = 0.5

        # Primary mechanic gets higher confidence
        if mechanic == context.mechanics_profile.primary_mechanic:
            base_confidence += 0.3

        # Add confidence based on mechanic-specific evidence
        if mechanic in context.mechanics_profile.mechanic_confidence:
            mechanic_confidence = context.mechanics_profile.mechanic_confidence[mechanic]
            base_confidence += mechanic_confidence * 0.3

        # Add confidence based on visual patterns
        relevant_patterns = [
            p for p in context.mechanics_profile.visual_patterns
            if self._pattern_supports_mechanic(p, mechanic)
        ]
        pattern_confidence = min(len(relevant_patterns) * 0.1, 0.2)
        base_confidence += pattern_confidence

        # Add confidence based on complexity match
        complexity = context.mechanics_profile.complexity_score
        if mechanic in [GameMechanic.SPATIAL_PUZZLE, GameMechanic.PHYSICS_SIMULATION]:
            # These mechanics benefit from higher complexity
            if complexity > 0.6:
                base_confidence += 0.1
        else:
            # Other mechanics might work better with lower complexity
            if complexity < 0.7:
                base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _pattern_supports_mechanic(self, pattern, mechanic: GameMechanic) -> bool:
        """Check if a visual pattern supports a specific mechanic."""
        support_map = {
            GameMechanic.PATTERN_COMPLETION: ["rectangle", "row_sequence", "col_sequence"],
            GameMechanic.OBJECT_MANIPULATION: ["color_region", "rectangle"],
            GameMechanic.PHYSICS_SIMULATION: ["circle", "color_region"],
            GameMechanic.SPATIAL_PUZZLE: ["rectangle", "circle"],
            GameMechanic.SORTING_ORGANIZING: ["color_region"],
            GameMechanic.NAVIGATION_PATHFINDING: ["rectangle", "color_region"]
        }

        return pattern.pattern_type in support_map.get(mechanic, [])

    def _generate_reasoning(self, context: HypothesisContext, mechanic: GameMechanic,
                          strategy: str, coordinates: List[Tuple[int, int]]) -> str:
        """Generate human-readable reasoning for the hypothesis."""

        reasoning_templates = {
            GameMechanic.PATTERN_COMPLETION: {
                "complete_missing_elements": "Pattern analysis detected incomplete grid structures with {grid_confidence:.1%} confidence. "
                                           "The {pattern_count} visual patterns suggest missing elements that can be completed by "
                                           "interacting with coordinates {sample_coords}.",
                "extend_sequences": "Detected repetitive sequences in the game grid. Strategy involves extending these "
                                  "patterns to their logical conclusion at predicted coordinates.",
                "fill_symmetrical_gaps": "Symmetry analysis shows {symmetry_type} symmetrical patterns with gaps. "
                                       "Filling these gaps should complete the pattern."
            },
            GameMechanic.OBJECT_MANIPULATION: {
                "move_discrete_objects": "Identified {object_count} discrete objects that can be manipulated. "
                                       "Objects appear to have clear boundaries and can be repositioned to {sample_coords}.",
                "transform_shapes": "Shape analysis indicates objects with transformable properties. "
                                  "Rotation or scaling may be required to solve the puzzle.",
                "reposition_elements": "Elements appear moveable based on spatial analysis. "
                                     "Repositioning to optimal locations should achieve the goal."
            },
            GameMechanic.PHYSICS_SIMULATION: {
                "simulate_gravity": "Detected objects that appear subject to gravity effects. "
                                  "Simulating falling objects should reveal the solution pattern.",
                "predict_collisions": "Object trajectories suggest collision points at coordinates {sample_coords}. "
                                    "Understanding collision outcomes is key to solving this puzzle.",
                "trace_movement_paths": "Movement patterns indicate specific trajectories. "
                                      "Following these paths should lead to the solution."
            },
            GameMechanic.SPATIAL_PUZZLE: {
                "fit_pieces_together": "Geometric analysis shows {shape_count} shapes with complexity {complexity:.1f}. "
                                     "These pieces appear designed to fit together at specific positions.",
                "rotate_objects": "Shapes require rotation to achieve proper spatial alignment. "
                                "Testing rotational positions at {sample_coords} should reveal the solution.",
                "spatial_alignment": "Spatial relationships suggest specific alignment requirements. "
                                   "Proper positioning is critical for solving this puzzle."
            },
            GameMechanic.SORTING_ORGANIZING: {
                "group_by_properties": "Color analysis detected {color_count} distinct colors with organized distribution. "
                                     "Grouping objects by properties should reveal the solution pattern.",
                "sort_sequences": "Sequential patterns detected that require proper ordering. "
                                "Sorting according to identified rules should solve the puzzle.",
                "organize_patterns": "Pattern organization is key. Arranging elements according to "
                                   "detected organizational principles should achieve the goal."
            },
            GameMechanic.NAVIGATION_PATHFINDING: {
                "find_optimal_path": "Grid structure suggests pathfinding challenge. "
                                   "Finding optimal route between endpoints at {sample_coords} is required.",
                "avoid_obstacles": "Obstacle patterns detected in grid. Navigation must account for "
                                 "obstacle avoidance while reaching the destination.",
                "connect_endpoints": "Clear endpoints identified. Challenge is to establish connection "
                                   "through the available grid space."
            }
        }

        template = reasoning_templates.get(mechanic, {}).get(strategy,
            "Strategy {strategy} applied to {mechanic} based on pattern analysis.")

        # Fill in template variables
        format_data = {
            "strategy": strategy,
            "mechanic": mechanic.value,
            "sample_coords": str(coordinates[:3]) if coordinates else "unknown",
            "grid_confidence": context.screenshot_features.get("cell_uniformity", 0.5),
            "pattern_count": len(context.mechanics_profile.visual_patterns),
            "object_count": len([p for p in context.mechanics_profile.visual_patterns if p.pattern_type == "color_region"]),
            "shape_count": context.screenshot_features.get("shape_count", 0),
            "complexity": context.mechanics_profile.complexity_score,
            "color_count": context.screenshot_features.get("color_count", 0),
            "symmetry_type": self._get_symmetry_types(context.screenshot_features)
        }

        try:
            return template.format(**format_data)
        except KeyError:
            # Fallback if template formatting fails
            return f"Apply {strategy} strategy for {mechanic.value} mechanic based on pattern analysis."

    def _get_symmetry_types(self, features: Dict[str, Any]) -> str:
        """Get human-readable symmetry types from features."""
        symmetry = features.get("symmetry_detected", {})
        types = []

        if symmetry.get("horizontal", False):
            types.append("horizontal")
        if symmetry.get("vertical", False):
            types.append("vertical")
        if symmetry.get("diagonal", False):
            types.append("diagonal")

        return ", ".join(types) if types else "no"

    def _collect_supporting_evidence(self, context: HypothesisContext,
                                   mechanic: GameMechanic, strategy: str) -> Dict[str, Any]:
        """Collect supporting evidence for the hypothesis."""
        evidence = {
            "visual_patterns": len(context.mechanics_profile.visual_patterns),
            "mechanic_confidence": context.mechanics_profile.mechanic_confidence.get(mechanic, 0.0),
            "complexity_score": context.mechanics_profile.complexity_score,
            "grid_detected": context.screenshot_features.get("grid_detected", False),
            "color_count": context.screenshot_features.get("color_count", 0),
            "shape_count": context.screenshot_features.get("shape_count", 0),
            "strategy_applied": strategy,
            "analysis_timestamp": datetime.now().isoformat()
        }

        # Add mechanic-specific evidence
        if mechanic == GameMechanic.PATTERN_COMPLETION:
            evidence.update({
                "symmetry_detected": context.screenshot_features.get("symmetry_detected", {}),
                "repetition_detected": context.screenshot_features.get("repetition_detected", {}),
                "cell_uniformity": context.screenshot_features.get("cell_uniformity", 0.0)
            })
        elif mechanic == GameMechanic.OBJECT_MANIPULATION:
            evidence.update({
                "discrete_objects": len([p for p in context.mechanics_profile.visual_patterns
                                       if p.pattern_type == "color_region"]),
                "object_boundaries": "clear" if context.screenshot_features.get("shape_count", 0) > 1 else "unclear"
            })
        elif mechanic == GameMechanic.PHYSICS_SIMULATION:
            evidence.update({
                "circular_objects": len([p for p in context.mechanics_profile.visual_patterns
                                       if p.pattern_type == "circle"]),
                "gradient_strength": context.screenshot_features.get("gradient_detected", {}).get("gradient_strength", 0)
            })

        return evidence

    def _map_mechanic_to_hypothesis_type(self, mechanic: GameMechanic) -> HypothesisType:
        """Map game mechanic to hypothesis type."""
        mapping = {
            GameMechanic.PATTERN_COMPLETION: HypothesisType.PATTERN_COMPLETION,
            GameMechanic.OBJECT_MANIPULATION: HypothesisType.OBJECT_MANIPULATION,
            GameMechanic.PHYSICS_SIMULATION: HypothesisType.PHYSICS_SIMULATION,
            GameMechanic.SPATIAL_PUZZLE: HypothesisType.SPATIAL_REASONING,
            GameMechanic.SORTING_ORGANIZING: HypothesisType.COLOR_MATCHING,
            GameMechanic.NAVIGATION_PATHFINDING: HypothesisType.NAVIGATION_PATH,
            GameMechanic.COLOR_MATCHING: HypothesisType.COLOR_MATCHING,
            GameMechanic.SHAPE_TRANSFORMATION: HypothesisType.SHAPE_TRANSFORMATION
        }

        return mapping.get(mechanic, HypothesisType.COORDINATE_SEQUENCE)

    def _generate_database_hypotheses(self, context: HypothesisContext) -> List[Hypothesis]:
        """Generate hypotheses by retrieving successful strategies from database."""
        hypotheses = []

        if not self.db_connection:
            logger.warning("No database connection available for hypothesis retrieval")
            return hypotheses

        try:
            # Retrieve successful hypotheses for this game type
            successful_hypotheses = self._retrieve_successful_hypotheses(context.game_type)

            for i, db_hypothesis in enumerate(successful_hypotheses[:3]):  # Top 3 from database
                hypothesis = self._adapt_database_hypothesis(context, db_hypothesis, i)
                if hypothesis:
                    hypotheses.append(hypothesis)

        except Exception as e:
            logger.error(f"Error retrieving database hypotheses: {e}")

        return hypotheses

    def _retrieve_successful_hypotheses(self, game_type: str) -> List[Dict[str, Any]]:
        """Retrieve successful hypotheses from database for a game type."""
        if not self.db_connection:
            return []

        try:
            cursor = self.db_connection.execute("""
                SELECT hypothesis_data, success_rate, test_count, reasoning
                FROM game_hypotheses
                WHERE game_type = ? AND success_rate > 0.6 AND test_count >= 3
                ORDER BY success_rate DESC, test_count DESC
                LIMIT 5
            """, (game_type,))

            results = []
            for row in cursor.fetchall():
                try:
                    hypothesis_data = json.loads(row[0])
                    hypothesis_data.update({
                        'success_rate': row[1],
                        'test_count': row[2],
                        'reasoning': row[3]
                    })
                    results.append(hypothesis_data)
                except json.JSONDecodeError:
                    continue

            return results

        except Exception as e:
            logger.error(f"Error querying successful hypotheses: {e}")
            return []

    def _adapt_database_hypothesis(self, context: HypothesisContext,
                                 db_hypothesis: Dict[str, Any], priority: int) -> Optional[Hypothesis]:
        """Adapt a database hypothesis to current game context."""

        try:
            # Extract information from database hypothesis
            predicted_coords = db_hypothesis.get('predicted_coordinates', [])
            expected_actions = db_hypothesis.get('expected_actions', [])
            description = db_hypothesis.get('description', 'Database-retrieved strategy')
            reasoning = db_hypothesis.get('reasoning', 'Based on successful past performance')

            # Adapt coordinates to current context if needed
            adapted_coords = self._adapt_coordinates_to_context(predicted_coords, context)

            # Calculate confidence based on past success
            base_confidence = min(db_hypothesis.get('success_rate', 0.5), 0.9)
            context_adaptation_score = self._calculate_context_adaptation_score(context, db_hypothesis)
            final_confidence = base_confidence * context_adaptation_score

            hypothesis_id = f"{context.game_id}_db_{priority}"

            return Hypothesis(
                hypothesis_id=hypothesis_id,
                hypothesis_type=HypothesisType(db_hypothesis.get('hypothesis_type', 'coordinate_sequence')),
                source=HypothesisSource.DATABASE_RETRIEVAL,
                description=f"Database strategy: {description}",
                predicted_coordinates=adapted_coords,
                expected_actions=expected_actions,
                confidence=final_confidence,
                reasoning=f"Retrieved from database with {db_hypothesis.get('success_rate', 0):.1%} success rate. {reasoning}",
                supporting_evidence={
                    "database_success_rate": db_hypothesis.get('success_rate', 0),
                    "database_test_count": db_hypothesis.get('test_count', 0),
                    "context_adaptation_score": context_adaptation_score,
                    "original_hypothesis": db_hypothesis
                },
                game_mechanics=[context.mechanics_profile.primary_mechanic],
                created_at=datetime.now(),
                tested=True,
                success_rate=db_hypothesis.get('success_rate', 0),
                test_count=db_hypothesis.get('test_count', 0)
            )

        except Exception as e:
            logger.error(f"Error adapting database hypothesis: {e}")
            return None

    def _adapt_coordinates_to_context(self, coords: List[Tuple[int, int]],
                                    context: HypothesisContext) -> List[Tuple[int, int]]:
        """Adapt coordinates from database to current game context."""
        if not coords:
            return []

        # For now, return coordinates as-is
        # In future, could implement intelligent coordinate scaling/adaptation
        return coords[:10]  # Limit to 10 coordinates

    def _calculate_context_adaptation_score(self, context: HypothesisContext,
                                          db_hypothesis: Dict[str, Any]) -> float:
        """Calculate how well a database hypothesis adapts to current context."""
        score = 1.0  # Start with perfect adaptation

        # Check if the game mechanics align
        db_mechanics = db_hypothesis.get('game_mechanics', [])
        current_mechanic = context.mechanics_profile.primary_mechanic.value

        if current_mechanic not in db_mechanics:
            score *= 0.7  # Penalty for mechanic mismatch

        # Check complexity alignment
        db_complexity = db_hypothesis.get('complexity_score', 0.5)
        current_complexity = context.mechanics_profile.complexity_score
        complexity_diff = abs(db_complexity - current_complexity)
        score *= max(0.5, 1.0 - complexity_diff)

        return score

    def _generate_knowledge_based_hypotheses(self, context: HypothesisContext) -> List[Hypothesis]:
        """Generate hypotheses based on game type knowledge."""
        hypotheses = []

        # Get recommendations from game type classifier
        try:
            recommendations = self.game_type_classifier.get_recommendations_for_game(context.game_id)

            # Convert recommendations to hypotheses
            hypothesis = self._create_knowledge_hypothesis(context, recommendations)
            if hypothesis:
                hypotheses.append(hypothesis)

        except Exception as e:
            logger.error(f"Error generating knowledge-based hypotheses: {e}")

        return hypotheses

    def _create_knowledge_hypothesis(self, context: HypothesisContext,
                                   recommendations: Dict[str, Any]) -> Optional[Hypothesis]:
        """Create hypothesis from game type knowledge recommendations."""

        try:
            # Extract recommended coordinates
            recommended_coords = recommendations.get('recommended_coordinates', [])[:5]

            # Extract button priorities as additional coordinates
            button_priorities = recommendations.get('button_priorities', [])[:3]
            priority_coords = [bp['coordinate'] for bp in button_priorities if 'coordinate' in bp]

            all_coords = recommended_coords + priority_coords

            if not all_coords:
                return None

            # Generate actions based on game type characteristics
            actions = ["click", "observe", "analyze"]
            if recommendations.get('is_action6_centric', False):
                actions = ["action6_click", "analyze_result", "adapt_strategy"]

            # Calculate confidence based on game type success rate
            base_confidence = recommendations.get('success_rate', 0.5)
            knowledge_confidence = min(base_confidence + 0.2, 0.9)  # Boost for knowledge-based

            hypothesis_id = f"{context.game_id}_knowledge"

            return Hypothesis(
                hypothesis_id=hypothesis_id,
                hypothesis_type=HypothesisType.COORDINATE_SEQUENCE,
                source=HypothesisSource.GAME_TYPE_KNOWLEDGE,
                description=f"Game type knowledge strategy for {context.game_type}",
                predicted_coordinates=all_coords,
                expected_actions=actions,
                confidence=knowledge_confidence,
                reasoning=f"Based on {context.game_type} game type knowledge with "
                         f"{recommendations.get('success_rate', 0):.1%} historical success rate. "
                         f"Action6-centric: {recommendations.get('is_action6_centric', False)}.",
                supporting_evidence={
                    "game_type_success_rate": recommendations.get('success_rate', 0),
                    "is_action6_centric": recommendations.get('is_action6_centric', False),
                    "action6_centric_count": recommendations.get('action6_centric_count', 0),
                    "similar_game_types": recommendations.get('similar_game_types', []),
                    "button_priorities_count": len(button_priorities),
                    "knowledge_source": "game_type_classifier"
                },
                game_mechanics=[context.mechanics_profile.primary_mechanic],
                created_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error creating knowledge hypothesis: {e}")
            return None

    def _generate_hybrid_hypotheses(self, context: HypothesisContext,
                                   existing_hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Generate hybrid hypotheses combining multiple sources."""
        hypotheses = []

        if len(existing_hypotheses) < 2:
            return hypotheses  # Need at least 2 hypotheses to combine

        # Combine top pattern and database hypotheses
        pattern_hypotheses = [h for h in existing_hypotheses if h.source == HypothesisSource.PATTERN_ANALYSIS]
        db_hypotheses = [h for h in existing_hypotheses if h.source == HypothesisSource.DATABASE_RETRIEVAL]
        knowledge_hypotheses = [h for h in existing_hypotheses if h.source == HypothesisSource.GAME_TYPE_KNOWLEDGE]

        # Create hybrid combinations
        if pattern_hypotheses and db_hypotheses:
            hybrid = self._combine_hypotheses(context, pattern_hypotheses[0], db_hypotheses[0], "pattern_db")
            if hybrid:
                hypotheses.append(hybrid)

        if pattern_hypotheses and knowledge_hypotheses:
            hybrid = self._combine_hypotheses(context, pattern_hypotheses[0], knowledge_hypotheses[0], "pattern_knowledge")
            if hybrid:
                hypotheses.append(hybrid)

        return hypotheses

    def _combine_hypotheses(self, context: HypothesisContext,
                          hyp1: Hypothesis, hyp2: Hypothesis, combination_type: str) -> Optional[Hypothesis]:
        """Combine two hypotheses into a hybrid hypothesis."""

        try:
            # Combine coordinates (remove duplicates, prioritize by confidence)
            all_coords = hyp1.predicted_coordinates + hyp2.predicted_coordinates
            unique_coords = []
            seen = set()

            for coord in all_coords:
                if coord not in seen:
                    unique_coords.append(coord)
                    seen.add(coord)

            # Limit to top 8 coordinates
            combined_coords = unique_coords[:8]

            # Combine actions
            combined_actions = list(set(hyp1.expected_actions + hyp2.expected_actions))

            # Calculate hybrid confidence (weighted average)
            weight1 = hyp1.confidence
            weight2 = hyp2.confidence
            total_weight = weight1 + weight2

            if total_weight > 0:
                hybrid_confidence = (hyp1.confidence * weight1 + hyp2.confidence * weight2) / total_weight
            else:
                hybrid_confidence = 0.5

            # Boost confidence slightly for hybrid approach
            hybrid_confidence = min(hybrid_confidence + 0.1, 1.0)

            # Combine reasoning
            combined_reasoning = f"Hybrid approach combining {hyp1.source.value} and {hyp2.source.value}: " \
                               f"{hyp1.reasoning[:100]}... COMBINED WITH {hyp2.reasoning[:100]}..."

            # Combine evidence
            combined_evidence = {
                "combination_type": combination_type,
                "source1": hyp1.source.value,
                "source2": hyp2.source.value,
                "confidence1": hyp1.confidence,
                "confidence2": hyp2.confidence,
                "hypothesis1_id": hyp1.hypothesis_id,
                "hypothesis2_id": hyp2.hypothesis_id,
                "evidence1": hyp1.supporting_evidence,
                "evidence2": hyp2.supporting_evidence
            }

            hypothesis_id = f"{context.game_id}_hybrid_{combination_type}"

            return Hypothesis(
                hypothesis_id=hypothesis_id,
                hypothesis_type=hyp1.hypothesis_type,  # Use primary hypothesis type
                source=HypothesisSource.HYBRID_APPROACH,
                description=f"Hybrid strategy: {hyp1.description} + {hyp2.description}",
                predicted_coordinates=combined_coords,
                expected_actions=combined_actions,
                confidence=hybrid_confidence,
                reasoning=combined_reasoning,
                supporting_evidence=combined_evidence,
                game_mechanics=list(set(hyp1.game_mechanics + hyp2.game_mechanics)),
                created_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error combining hypotheses: {e}")
            return None

    def _score_and_rank_hypotheses(self, hypotheses: List[Hypothesis],
                                 context: HypothesisContext) -> List[Hypothesis]:
        """Score and rank hypotheses by overall quality and relevance."""

        for hypothesis in hypotheses:
            score = self._calculate_hypothesis_score(hypothesis, context)
            # Store score in confidence field (already calculated, but we can adjust)
            hypothesis.confidence = min(hypothesis.confidence * score, 1.0)

        # Sort by confidence (higher is better)
        ranked_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

        return ranked_hypotheses

    def _calculate_hypothesis_score(self, hypothesis: Hypothesis, context: HypothesisContext) -> float:
        """Calculate overall quality score for a hypothesis."""
        score = 1.0

        # Source reliability scoring
        source_scores = {
            HypothesisSource.DATABASE_RETRIEVAL: 1.2,  # Boost for proven strategies
            HypothesisSource.HYBRID_APPROACH: 1.1,     # Boost for combined approaches
            HypothesisSource.GAME_TYPE_KNOWLEDGE: 1.0,
            HypothesisSource.PATTERN_ANALYSIS: 0.9
        }
        score *= source_scores.get(hypothesis.source, 1.0)

        # Coordinate quality scoring
        if len(hypothesis.predicted_coordinates) > 0:
            score *= 1.1  # Boost for having coordinates
            if len(hypothesis.predicted_coordinates) > 10:
                score *= 0.9  # Slight penalty for too many coordinates
        else:
            score *= 0.7  # Penalty for no coordinates

        # Action sequence quality
        if len(hypothesis.expected_actions) >= 2:
            score *= 1.05  # Slight boost for multi-step actions

        # Evidence quality
        evidence_count = len(hypothesis.supporting_evidence)
        if evidence_count > 5:
            score *= 1.1  # Boost for rich evidence
        elif evidence_count < 3:
            score *= 0.9  # Slight penalty for sparse evidence

        return min(score, 1.5)  # Cap the scoring multiplier

    def _get_historical_successes(self, game_type: str) -> List[Dict[str, Any]]:
        """Get historical successful strategies for a game type."""
        if not self.db_connection:
            return []

        try:
            cursor = self.db_connection.execute("""
                SELECT hypothesis_data, success_rate, test_count
                FROM game_hypotheses
                WHERE game_type = ? AND success_rate > 0.5
                ORDER BY success_rate DESC, test_count DESC
                LIMIT 10
            """, (game_type,))

            results = []
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[0])
                    data.update({'success_rate': row[1], 'test_count': row[2]})
                    results.append(data)
                except json.JSONDecodeError:
                    continue

            return results

        except Exception as e:
            logger.error(f"Error retrieving historical successes: {e}")
            return []

    def _get_similar_games_data(self, game_type: str) -> List[Dict[str, Any]]:
        """Get data from similar game types."""
        if not self.db_connection:
            return []

        try:
            # Get similar game types from classifier
            similar_types = self.game_type_classifier.get_similar_game_types(game_type)

            all_data = []
            for similar_type in similar_types[:5]:  # Limit to top 5 similar types
                data = self._get_historical_successes(similar_type)
                all_data.extend(data)

            return all_data[:15]  # Limit total results

        except Exception as e:
            logger.error(f"Error retrieving similar games data: {e}")
            return []

    def save_hypothesis_result(self, hypothesis: Hypothesis, success: bool,
                             score_change: float = 0.0, additional_data: Dict[str, Any] = None):
        """Save the result of testing a hypothesis."""
        hypothesis.tested = True
        hypothesis.test_count += 1

        # Update success rate
        if hypothesis.test_count == 1:
            hypothesis.success_rate = 1.0 if success else 0.0
        else:
            # Weighted average with previous results
            old_successes = hypothesis.success_rate * (hypothesis.test_count - 1)
            new_successes = old_successes + (1.0 if success else 0.0)
            hypothesis.success_rate = new_successes / hypothesis.test_count

        # Save to database if available
        if self.db_connection:
            self._save_hypothesis_to_database(hypothesis, success, score_change, additional_data)

        logger.info(f"Hypothesis {hypothesis.hypothesis_id} tested: Success={success}, "
                   f"New success rate: {hypothesis.success_rate:.2%}")

    def _save_hypothesis_to_database(self, hypothesis: Hypothesis, success: bool,
                                   score_change: float, additional_data: Dict[str, Any]):
        """Save hypothesis test result to database."""
        try:
            # Prepare hypothesis data for storage
            hypothesis_data = {
                'hypothesis_type': hypothesis.hypothesis_type.value,
                'description': hypothesis.description,
                'predicted_coordinates': hypothesis.predicted_coordinates,
                'expected_actions': hypothesis.expected_actions,
                'game_mechanics': [m.value for m in hypothesis.game_mechanics],
                'complexity_score': getattr(hypothesis, 'complexity_score', 0.5),
                'additional_data': additional_data or {}
            }

            # Extract game type from hypothesis context
            game_type = hypothesis.hypothesis_id.split('_')[1] if '_' in hypothesis.hypothesis_id else 'unknown'

            self.db_connection.execute("""
                INSERT OR REPLACE INTO game_hypotheses
                (hypothesis_id, game_type, hypothesis_data, confidence, success_rate, test_count,
                 reasoning, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hypothesis.hypothesis_id,
                game_type,
                json.dumps(hypothesis_data),
                hypothesis.confidence,
                hypothesis.success_rate,
                hypothesis.test_count,
                hypothesis.reasoning,
                hypothesis.created_at.isoformat(),
                datetime.now().isoformat()
            ))

            self.db_connection.commit()
            logger.debug(f"Saved hypothesis {hypothesis.hypothesis_id} to database")

        except Exception as e:
            logger.error(f"Error saving hypothesis to database: {e}")

# Module initialization
def create_hypothesis_generator(db_connection: Optional[sqlite3.Connection] = None) -> HypothesisGenerator:
    """Factory function to create a HypothesisGenerator instance."""
    return HypothesisGenerator(db_connection)

# Singleton instance for global use
_generator_instance = None

def get_hypothesis_generator(db_connection: Optional[sqlite3.Connection] = None) -> HypothesisGenerator:
    """Get the singleton HypothesisGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = create_hypothesis_generator(db_connection)
    return _generator_instance