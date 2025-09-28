"""
Bayesian Inference Engine - Tier 3 System

Provides probabilistic reasoning about game mechanics, hypothesis testing,
and uncertainty-aware decision support using Bayesian networks and belief updating.

Key Features:
- Bayesian Networks for action-outcome relationships
- Hypothesis tracking and validation with confidence intervals
- Probabilistic game mechanics modeling
- Evidence accumulation and belief updating
- Uncertainty quantification for decision support
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import math
from collections import defaultdict, deque
import sqlite3

logger = logging.getLogger(__name__)


class HypothesisType(Enum):
    """Types of hypotheses the system can track."""
    ACTION_OUTCOME = "action_outcome"  # Action X leads to outcome Y
    SEQUENCE_PATTERN = "sequence_pattern"  # Sequence ABC leads to result Z
    CONDITIONAL_RULE = "conditional_rule"  # If condition X, then action Y works
    COORDINATE_EFFECTIVENESS = "coordinate_effectiveness"  # Coordinate (x,y) is effective in context Z
    TIMING_DEPENDENCY = "timing_dependency"  # Action timing affects outcomes
    STATE_TRANSITION = "state_transition"  # Game state transitions follow pattern X
    MULTI_STEP_STRATEGY = "multi_step_strategy"  # Complex strategies with multiple components


class EvidenceType(Enum):
    """Types of evidence that can support or refute hypotheses."""
    DIRECT_OBSERVATION = "direct_observation"  # Direct action-outcome observation
    PATTERN_MATCH = "pattern_match"  # Pattern matching previous successful cases
    STATISTICAL_CORRELATION = "statistical_correlation"  # Statistical relationship
    CONTEXTUAL_SIMILARITY = "contextual_similarity"  # Similar contexts showing similar results
    NEGATIVE_EVIDENCE = "negative_evidence"  # Evidence against the hypothesis


@dataclass
class BayesianHypothesis:
    """Represents a hypothesis with Bayesian probability tracking."""
    hypothesis_id: str
    hypothesis_type: HypothesisType
    description: str
    prior_probability: float
    posterior_probability: float
    evidence_count: int
    supporting_evidence: int
    refuting_evidence: int
    confidence_interval: Tuple[float, float]
    created_at: datetime
    last_updated: datetime
    context_conditions: Dict[str, Any]
    validation_threshold: float

    def update_probability(self, evidence_support: bool, evidence_strength: float = 1.0) -> None:
        """Update posterior probability using Bayesian updating."""
        # Simple Bayesian update using evidence strength
        if evidence_support:
            self.supporting_evidence += 1
            # Increase probability based on evidence strength
            likelihood_ratio = 1.0 + evidence_strength
        else:
            self.refuting_evidence += 1
            # Decrease probability based on evidence strength
            likelihood_ratio = 1.0 / (1.0 + evidence_strength)

        self.evidence_count += 1

        # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        # Simplified using likelihood ratio
        odds = self.posterior_probability / (1 - self.posterior_probability)
        new_odds = odds * likelihood_ratio
        self.posterior_probability = new_odds / (1 + new_odds)

        # Ensure probability stays in valid range
        self.posterior_probability = max(0.001, min(0.999, self.posterior_probability))

        # Update confidence interval (simplified)
        n = self.evidence_count
        if n > 5:  # Need minimum evidence for meaningful confidence interval
            std_error = math.sqrt(self.posterior_probability * (1 - self.posterior_probability) / n)
            margin = 1.96 * std_error  # 95% confidence interval
            self.confidence_interval = (
                max(0.0, self.posterior_probability - margin),
                min(1.0, self.posterior_probability + margin)
            )

        self.last_updated = datetime.now()


@dataclass
class Evidence:
    """Represents a piece of evidence for or against a hypothesis."""
    evidence_id: str
    hypothesis_id: str
    evidence_type: EvidenceType
    supports_hypothesis: bool
    strength: float  # 0.0 to 1.0
    context: Dict[str, Any]
    observed_at: datetime
    game_id: str
    session_id: str


@dataclass
class BayesianPrediction:
    """Represents a probabilistic prediction about an action or outcome."""
    prediction_id: str
    predicted_action: Dict[str, Any]
    success_probability: float
    confidence_level: float
    supporting_hypotheses: List[str]
    uncertainty_factors: List[str]
    evidence_summary: Dict[str, int]
    context_conditions: Dict[str, Any]
    created_at: datetime


@dataclass
class BeliefNetwork:
    """Represents a network of beliefs about game mechanics."""
    network_id: str
    nodes: Dict[str, Dict[str, Any]]  # Node ID -> node properties
    edges: List[Tuple[str, str, float]]  # (from_node, to_node, strength)
    conditional_probabilities: Dict[str, Dict[str, float]]
    evidence_propagation_rules: Dict[str, List[str]]
    last_updated: datetime


class BayesianInferenceEngine:
    """
    Advanced Bayesian Inference Engine for probabilistic reasoning about game mechanics.

    This system builds and maintains probabilistic models of game behavior,
    tracks hypotheses about game rules, and provides uncertainty-aware decision support.
    """

    def __init__(self, db_connection: sqlite3.Connection):
        """Initialize the Bayesian Inference Engine."""
        self.db_connection = db_connection
        self.hypotheses: Dict[str, BayesianHypothesis] = {}
        self.evidence_store: Dict[str, List[Evidence]] = defaultdict(list)
        self.belief_networks: Dict[str, BeliefNetwork] = {}
        self.prediction_cache: Dict[str, BayesianPrediction] = {}

        # Configuration parameters
        self.max_hypotheses_per_type = 100
        self.evidence_decay_days = 30
        self.min_evidence_for_prediction = 3
        self.confidence_threshold = 0.7

        # Initialize database tables
        self._init_database_schema()

        # Load existing data
        asyncio.create_task(self._load_existing_data())

        logger.info("Bayesian Inference Engine initialized")

    def _init_database_schema(self) -> None:
        """Initialize database schema for Bayesian inference."""
        try:
            cursor = self.db_connection.cursor()

            # Hypotheses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bayesian_hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    hypothesis_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    prior_probability REAL NOT NULL,
                    posterior_probability REAL NOT NULL,
                    evidence_count INTEGER DEFAULT 0,
                    supporting_evidence INTEGER DEFAULT 0,
                    refuting_evidence INTEGER DEFAULT 0,
                    confidence_lower REAL DEFAULT 0.0,
                    confidence_upper REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    context_conditions TEXT, -- JSON
                    validation_threshold REAL DEFAULT 0.8,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            # Evidence table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bayesian_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    supports_hypothesis BOOLEAN NOT NULL,
                    strength REAL NOT NULL,
                    context TEXT, -- JSON
                    observed_at TEXT NOT NULL,
                    game_id TEXT,
                    session_id TEXT,
                    FOREIGN KEY (hypothesis_id) REFERENCES bayesian_hypotheses (hypothesis_id)
                )
            """)

            # Belief networks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bayesian_belief_networks (
                    network_id TEXT PRIMARY KEY,
                    game_type TEXT,
                    nodes TEXT NOT NULL, -- JSON
                    edges TEXT NOT NULL, -- JSON
                    conditional_probabilities TEXT, -- JSON
                    evidence_propagation_rules TEXT, -- JSON
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    accuracy_score REAL DEFAULT 0.5
                )
            """)

            # Predictions table for tracking accuracy
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bayesian_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    predicted_action TEXT NOT NULL, -- JSON
                    success_probability REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    supporting_hypotheses TEXT, -- JSON
                    uncertainty_factors TEXT, -- JSON
                    evidence_summary TEXT, -- JSON
                    context_conditions TEXT, -- JSON
                    created_at TEXT NOT NULL,
                    actual_outcome TEXT, -- JSON (filled after observation)
                    prediction_accuracy REAL, -- 0.0 to 1.0 (filled after observation)
                    game_id TEXT,
                    session_id TEXT
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hypotheses_type ON bayesian_hypotheses (hypothesis_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_hypothesis ON bayesian_evidence (hypothesis_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_game ON bayesian_evidence (game_id, session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_game ON bayesian_predictions (game_id, session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_networks_game_type ON bayesian_belief_networks (game_type)")

            self.db_connection.commit()
            logger.info("Bayesian inference database schema initialized")

        except Exception as e:
            logger.error(f"Error initializing Bayesian inference database schema: {e}")
            raise

    async def _load_existing_data(self) -> None:
        """Load existing hypotheses and belief networks from database."""
        try:
            cursor = self.db_connection.cursor()

            # Load hypotheses
            cursor.execute("""
                SELECT * FROM bayesian_hypotheses WHERE is_active = 1
                ORDER BY last_updated DESC
            """)

            for row in cursor.fetchall():
                hypothesis = BayesianHypothesis(
                    hypothesis_id=row[0],
                    hypothesis_type=HypothesisType(row[1]),
                    description=row[2],
                    prior_probability=row[3],
                    posterior_probability=row[4],
                    evidence_count=row[5],
                    supporting_evidence=row[6],
                    refuting_evidence=row[7],
                    confidence_interval=(row[8], row[9]),
                    created_at=datetime.fromisoformat(row[10]),
                    last_updated=datetime.fromisoformat(row[11]),
                    context_conditions=json.loads(row[12]) if row[12] else {},
                    validation_threshold=row[13]
                )
                self.hypotheses[hypothesis.hypothesis_id] = hypothesis

            # Load evidence for each hypothesis
            for hypothesis_id in self.hypotheses.keys():
                cursor.execute("""
                    SELECT * FROM bayesian_evidence
                    WHERE hypothesis_id = ?
                    ORDER BY observed_at DESC
                """, (hypothesis_id,))

                for evidence_row in cursor.fetchall():
                    evidence = Evidence(
                        evidence_id=evidence_row[0],
                        hypothesis_id=evidence_row[1],
                        evidence_type=EvidenceType(evidence_row[2]),
                        supports_hypothesis=bool(evidence_row[3]),
                        strength=evidence_row[4],
                        context=json.loads(evidence_row[5]) if evidence_row[5] else {},
                        observed_at=datetime.fromisoformat(evidence_row[6]),
                        game_id=evidence_row[7],
                        session_id=evidence_row[8]
                    )
                    self.evidence_store[hypothesis_id].append(evidence)

            # Load belief networks
            cursor.execute("SELECT * FROM bayesian_belief_networks ORDER BY last_updated DESC")
            for network_row in cursor.fetchall():
                network = BeliefNetwork(
                    network_id=network_row[0],
                    nodes=json.loads(network_row[2]),
                    edges=json.loads(network_row[3]),
                    conditional_probabilities=json.loads(network_row[4]),
                    evidence_propagation_rules=json.loads(network_row[5]),
                    last_updated=datetime.fromisoformat(network_row[7])
                )
                self.belief_networks[network.network_id] = network

            logger.info(f"Loaded {len(self.hypotheses)} hypotheses and {len(self.belief_networks)} belief networks")

        except Exception as e:
            logger.error(f"Error loading existing Bayesian data: {e}")

    async def create_hypothesis(self,
                              hypothesis_type: HypothesisType,
                              description: str,
                              prior_probability: float,
                              context_conditions: Dict[str, Any],
                              game_id: str,
                              session_id: str) -> str:
        """Create a new hypothesis for Bayesian tracking."""
        try:
            hypothesis_id = f"{hypothesis_type.value}_{game_id}_{len(self.hypotheses)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            hypothesis = BayesianHypothesis(
                hypothesis_id=hypothesis_id,
                hypothesis_type=hypothesis_type,
                description=description,
                prior_probability=prior_probability,
                posterior_probability=prior_probability,  # Start with prior
                evidence_count=0,
                supporting_evidence=0,
                refuting_evidence=0,
                confidence_interval=(0.0, 1.0),  # Wide initial interval
                created_at=datetime.now(),
                last_updated=datetime.now(),
                context_conditions=context_conditions,
                validation_threshold=0.8
            )

            # Store in memory
            self.hypotheses[hypothesis_id] = hypothesis

            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO bayesian_hypotheses (
                    hypothesis_id, hypothesis_type, description, prior_probability,
                    posterior_probability, evidence_count, supporting_evidence,
                    refuting_evidence, confidence_lower, confidence_upper,
                    created_at, last_updated, context_conditions, validation_threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hypothesis_id, hypothesis_type.value, description, prior_probability,
                prior_probability, 0, 0, 0, 0.0, 1.0,
                hypothesis.created_at.isoformat(), hypothesis.last_updated.isoformat(),
                json.dumps(context_conditions), 0.8
            ))
            self.db_connection.commit()

            logger.info(f"Created hypothesis: {hypothesis_id} - {description}")
            return hypothesis_id

        except Exception as e:
            logger.error(f"Error creating hypothesis: {e}")
            raise

    async def add_evidence(self,
                          hypothesis_id: str,
                          evidence_type: EvidenceType,
                          supports_hypothesis: bool,
                          strength: float,
                          context: Dict[str, Any],
                          game_id: str,
                          session_id: str) -> str:
        """Add evidence for or against a hypothesis and update probabilities."""
        try:
            if hypothesis_id not in self.hypotheses:
                logger.warning(f"Hypothesis {hypothesis_id} not found")
                return None

            evidence_id = f"evidence_{hypothesis_id}_{len(self.evidence_store[hypothesis_id])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            evidence = Evidence(
                evidence_id=evidence_id,
                hypothesis_id=hypothesis_id,
                evidence_type=evidence_type,
                supports_hypothesis=supports_hypothesis,
                strength=strength,
                context=context,
                observed_at=datetime.now(),
                game_id=game_id,
                session_id=session_id
            )

            # Store evidence
            self.evidence_store[hypothesis_id].append(evidence)

            # Update hypothesis probability
            hypothesis = self.hypotheses[hypothesis_id]
            hypothesis.update_probability(supports_hypothesis, strength)

            # Store evidence in database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO bayesian_evidence (
                    evidence_id, hypothesis_id, evidence_type, supports_hypothesis,
                    strength, context, observed_at, game_id, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evidence_id, hypothesis_id, evidence_type.value, supports_hypothesis,
                strength, json.dumps(context), evidence.observed_at.isoformat(),
                game_id, session_id
            ))

            # Update hypothesis in database
            cursor.execute("""
                UPDATE bayesian_hypotheses SET
                    posterior_probability = ?, evidence_count = ?,
                    supporting_evidence = ?, refuting_evidence = ?,
                    confidence_lower = ?, confidence_upper = ?, last_updated = ?
                WHERE hypothesis_id = ?
            """, (
                hypothesis.posterior_probability, hypothesis.evidence_count,
                hypothesis.supporting_evidence, hypothesis.refuting_evidence,
                hypothesis.confidence_interval[0], hypothesis.confidence_interval[1],
                hypothesis.last_updated.isoformat(), hypothesis_id
            ))
            self.db_connection.commit()

            logger.debug(f"Added evidence to {hypothesis_id}: {supports_hypothesis} (strength: {strength})")
            return evidence_id

        except Exception as e:
            logger.error(f"Error adding evidence: {e}")
            raise

    async def generate_prediction(self,
                                action_candidates: List[Dict[str, Any]],
                                current_context: Dict[str, Any],
                                game_id: str,
                                session_id: str) -> Optional[BayesianPrediction]:
        """Generate probabilistic predictions for action candidates."""
        try:
            if not action_candidates:
                return None

            best_action = None
            best_probability = 0.0
            best_confidence = 0.0
            supporting_hypotheses = []
            uncertainty_factors = []
            evidence_summary = defaultdict(int)

            for action in action_candidates:
                action_probability = 0.0
                action_confidence = 0.0
                action_hypotheses = []

                # Evaluate action against all relevant hypotheses
                for hypothesis_id, hypothesis in self.hypotheses.items():
                    if not self._is_hypothesis_relevant(hypothesis, action, current_context):
                        continue

                    # Check if this hypothesis supports this action
                    support_strength = self._calculate_hypothesis_support(
                        hypothesis, action, current_context
                    )

                    if support_strength > 0:
                        # Weight by posterior probability and confidence
                        conf_width = hypothesis.confidence_interval[1] - hypothesis.confidence_interval[0]
                        confidence_factor = max(0.1, 1.0 - conf_width)  # Higher confidence = narrower interval

                        weighted_probability = (hypothesis.posterior_probability *
                                              support_strength * confidence_factor)
                        action_probability += weighted_probability
                        action_confidence += confidence_factor
                        action_hypotheses.append(hypothesis_id)

                        # Count evidence types
                        for evidence in self.evidence_store[hypothesis_id]:
                            evidence_summary[evidence.evidence_type.value] += 1

                # Normalize probabilities
                if len(action_hypotheses) > 0:
                    action_probability /= len(action_hypotheses)
                    action_confidence /= len(action_hypotheses)

                # Track uncertainty factors
                if action_confidence < self.confidence_threshold:
                    uncertainty_factors.append(f"Low confidence for action {action.get('id', 'unknown')}")

                if action_probability > best_probability:
                    best_action = action
                    best_probability = action_probability
                    best_confidence = action_confidence
                    supporting_hypotheses = action_hypotheses

            if best_action is None:
                return None

            # Additional uncertainty analysis
            if best_probability < 0.5:
                uncertainty_factors.append("Low overall success probability")
            if len(supporting_hypotheses) < self.min_evidence_for_prediction:
                uncertainty_factors.append("Insufficient supporting hypotheses")

            prediction_id = f"prediction_{game_id}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            prediction = BayesianPrediction(
                prediction_id=prediction_id,
                predicted_action=best_action,
                success_probability=best_probability,
                confidence_level=best_confidence,
                supporting_hypotheses=supporting_hypotheses,
                uncertainty_factors=uncertainty_factors,
                evidence_summary=dict(evidence_summary),
                context_conditions=current_context,
                created_at=datetime.now()
            )

            # Store prediction for later accuracy evaluation
            self.prediction_cache[prediction_id] = prediction

            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO bayesian_predictions (
                    prediction_id, predicted_action, success_probability,
                    confidence_level, supporting_hypotheses, uncertainty_factors,
                    evidence_summary, context_conditions, created_at, game_id, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id, json.dumps(best_action), best_probability,
                best_confidence, json.dumps(supporting_hypotheses),
                json.dumps(uncertainty_factors), json.dumps(dict(evidence_summary)),
                json.dumps(current_context), prediction.created_at.isoformat(),
                game_id, session_id
            ))
            self.db_connection.commit()

            return prediction

        except Exception as e:
            logger.error(f"Error generating Bayesian prediction: {e}")
            return None

    def _is_hypothesis_relevant(self,
                              hypothesis: BayesianHypothesis,
                              action: Dict[str, Any],
                              context: Dict[str, Any]) -> bool:
        """Check if a hypothesis is relevant to the current action and context."""
        try:
            # Check context conditions
            for key, expected_value in hypothesis.context_conditions.items():
                if key in context:
                    if isinstance(expected_value, (list, tuple)):
                        if context[key] not in expected_value:
                            return False
                    elif context[key] != expected_value:
                        return False

            # Check hypothesis type relevance
            if hypothesis.hypothesis_type == HypothesisType.ACTION_OUTCOME:
                return 'id' in action
            elif hypothesis.hypothesis_type == HypothesisType.COORDINATE_EFFECTIVENESS:
                return action.get('id') == 6 and 'x' in action and 'y' in action
            elif hypothesis.hypothesis_type == HypothesisType.CONDITIONAL_RULE:
                return True  # Always potentially relevant

            return True

        except Exception as e:
            logger.error(f"Error checking hypothesis relevance: {e}")
            return False

    def _calculate_hypothesis_support(self,
                                    hypothesis: BayesianHypothesis,
                                    action: Dict[str, Any],
                                    context: Dict[str, Any]) -> float:
        """Calculate how much a hypothesis supports a particular action."""
        try:
            support_strength = 0.0

            if hypothesis.hypothesis_type == HypothesisType.ACTION_OUTCOME:
                # Simple action matching
                if 'action_id' in hypothesis.context_conditions:
                    if hypothesis.context_conditions['action_id'] == action.get('id'):
                        support_strength = 1.0

            elif hypothesis.hypothesis_type == HypothesisType.COORDINATE_EFFECTIVENESS:
                if action.get('id') == 6:
                    # Check coordinate proximity
                    hyp_x = hypothesis.context_conditions.get('x')
                    hyp_y = hypothesis.context_conditions.get('y')
                    action_x = action.get('x')
                    action_y = action.get('y')

                    if all(v is not None for v in [hyp_x, hyp_y, action_x, action_y]):
                        distance = math.sqrt((hyp_x - action_x)**2 + (hyp_y - action_y)**2)
                        # Closer coordinates get higher support
                        support_strength = max(0.0, 1.0 - (distance / 10.0))

            elif hypothesis.hypothesis_type == HypothesisType.CONDITIONAL_RULE:
                # Check if conditions are met
                condition_met = True
                for key, value in hypothesis.context_conditions.items():
                    if key.startswith('condition_'):
                        if context.get(key[10:]) != value:  # Remove 'condition_' prefix
                            condition_met = False
                            break

                if condition_met:
                    support_strength = 1.0

            return support_strength

        except Exception as e:
            logger.error(f"Error calculating hypothesis support: {e}")
            return 0.0

    async def update_prediction_accuracy(self,
                                       prediction_id: str,
                                       actual_outcome: Dict[str, Any]) -> None:
        """Update prediction accuracy after observing actual outcome."""
        try:
            if prediction_id not in self.prediction_cache:
                logger.warning(f"Prediction {prediction_id} not found in cache")
                return

            prediction = self.prediction_cache[prediction_id]

            # Calculate prediction accuracy
            predicted_success = prediction.success_probability > 0.5
            actual_success = actual_outcome.get('score_improvement', 0) > 0

            if predicted_success == actual_success:
                accuracy = prediction.confidence_level
            else:
                accuracy = 1.0 - prediction.confidence_level

            # Update database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE bayesian_predictions SET
                    actual_outcome = ?, prediction_accuracy = ?
                WHERE prediction_id = ?
            """, (json.dumps(actual_outcome), accuracy, prediction_id))
            self.db_connection.commit()

            # Use this feedback to improve hypothesis probabilities
            await self._learn_from_prediction_outcome(prediction, actual_outcome, accuracy)

        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}")

    async def _learn_from_prediction_outcome(self,
                                           prediction: BayesianPrediction,
                                           actual_outcome: Dict[str, Any],
                                           accuracy: float) -> None:
        """Learn from prediction outcomes to improve hypothesis probabilities."""
        try:
            outcome_positive = actual_outcome.get('score_improvement', 0) > 0

            # Update supporting hypotheses based on outcome
            for hypothesis_id in prediction.supporting_hypotheses:
                if hypothesis_id in self.hypotheses:
                    # If prediction was accurate, strengthen supporting hypotheses
                    # If prediction was wrong, weaken them
                    evidence_strength = accuracy if outcome_positive else (1.0 - accuracy)

                    await self.add_evidence(
                        hypothesis_id=hypothesis_id,
                        evidence_type=EvidenceType.DIRECT_OBSERVATION,
                        supports_hypothesis=outcome_positive,
                        strength=evidence_strength,
                        context=prediction.context_conditions,
                        game_id=actual_outcome.get('game_id', 'unknown'),
                        session_id=actual_outcome.get('session_id', 'unknown')
                    )

        except Exception as e:
            logger.error(f"Error learning from prediction outcome: {e}")

    async def get_hypothesis_insights(self, game_id: str) -> Dict[str, Any]:
        """Get insights about current hypotheses and their confidence levels."""
        try:
            insights = {
                'total_hypotheses': len(self.hypotheses),
                'high_confidence_hypotheses': [],
                'low_confidence_hypotheses': [],
                'recently_updated_hypotheses': [],
                'hypothesis_distribution': defaultdict(int),
                'evidence_summary': defaultdict(int)
            }

            current_time = datetime.now()

            for hypothesis in self.hypotheses.values():
                # Count by type
                insights['hypothesis_distribution'][hypothesis.hypothesis_type.value] += 1

                # Check confidence level
                conf_width = hypothesis.confidence_interval[1] - hypothesis.confidence_interval[0]
                confidence = max(0.0, 1.0 - conf_width)

                if confidence > 0.8:
                    insights['high_confidence_hypotheses'].append({
                        'id': hypothesis.hypothesis_id,
                        'description': hypothesis.description,
                        'probability': hypothesis.posterior_probability,
                        'confidence': confidence,
                        'evidence_count': hypothesis.evidence_count
                    })
                elif confidence < 0.3:
                    insights['low_confidence_hypotheses'].append({
                        'id': hypothesis.hypothesis_id,
                        'description': hypothesis.description,
                        'probability': hypothesis.posterior_probability,
                        'confidence': confidence,
                        'evidence_count': hypothesis.evidence_count
                    })

                # Check recent updates
                if (current_time - hypothesis.last_updated).total_seconds() < 3600:  # Last hour
                    insights['recently_updated_hypotheses'].append({
                        'id': hypothesis.hypothesis_id,
                        'description': hypothesis.description,
                        'last_updated': hypothesis.last_updated.isoformat()
                    })

                # Count evidence
                for evidence in self.evidence_store[hypothesis.hypothesis_id]:
                    insights['evidence_summary'][evidence.evidence_type.value] += 1

            return insights

        except Exception as e:
            logger.error(f"Error getting hypothesis insights: {e}")
            return {}

    async def prune_low_confidence_hypotheses(self) -> int:
        """Remove hypotheses with consistently low confidence or old evidence."""
        try:
            pruned_count = 0
            current_time = datetime.now()
            hypotheses_to_remove = []

            for hypothesis_id, hypothesis in self.hypotheses.items():
                should_prune = False

                # Prune if very low confidence and enough evidence
                conf_width = hypothesis.confidence_interval[1] - hypothesis.confidence_interval[0]
                confidence = max(0.0, 1.0 - conf_width)

                if (confidence < 0.1 and hypothesis.evidence_count > 10):
                    should_prune = True

                # Prune if very old and low probability
                days_old = (current_time - hypothesis.last_updated).days
                if (days_old > self.evidence_decay_days and
                    hypothesis.posterior_probability < 0.2):
                    should_prune = True

                # Prune if contradicted by strong evidence
                if (hypothesis.evidence_count > 5 and
                    hypothesis.refuting_evidence > hypothesis.supporting_evidence * 2):
                    should_prune = True

                if should_prune:
                    hypotheses_to_remove.append(hypothesis_id)

            # Remove pruned hypotheses
            for hypothesis_id in hypotheses_to_remove:
                del self.hypotheses[hypothesis_id]
                if hypothesis_id in self.evidence_store:
                    del self.evidence_store[hypothesis_id]

                # Mark as inactive in database
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    UPDATE bayesian_hypotheses SET is_active = 0
                    WHERE hypothesis_id = ?
                """, (hypothesis_id,))

                pruned_count += 1

            if pruned_count > 0:
                self.db_connection.commit()
                logger.info(f"Pruned {pruned_count} low-confidence hypotheses")

            return pruned_count

        except Exception as e:
            logger.error(f"Error pruning hypotheses: {e}")
            return 0

    def set_attention_coordination(self, attention_controller, communication_system) -> None:
        """Set attention controller and communication system for coordination."""
        self.attention_controller = attention_controller
        self.communication_system = communication_system
        logger.info("Bayesian inference engine linked with attention coordination")

    def set_fitness_evolution_coordination(self, fitness_evolution_system) -> None:
        """Set fitness evolution system for coordination."""
        self.fitness_evolution_system = fitness_evolution_system
        logger.info("Bayesian inference engine linked with fitness evolution")

    async def request_attention_allocation(self,
                                         game_id: str,
                                         session_id: str,
                                         reasoning_complexity: float) -> Optional[Dict[str, Any]]:
        """Request attention allocation for complex Bayesian reasoning."""
        try:
            if not hasattr(self, 'attention_controller') or not self.attention_controller:
                return None

            from src.core.central_attention_controller import SubsystemDemand

            bayesian_demand = SubsystemDemand(
                subsystem_name="bayesian_inference",
                requested_priority=min(0.8, 0.3 + reasoning_complexity),
                current_load=reasoning_complexity,
                processing_complexity=reasoning_complexity,
                urgency_level=3 if reasoning_complexity > 0.7 else 2,
                justification="Probabilistic reasoning and hypothesis updating required",
                context_data={
                    "active_hypotheses": len(self.hypotheses),
                    "reasoning_complexity": reasoning_complexity,
                    "session_id": session_id
                }
            )

            allocation = await self.attention_controller.allocate_attention_resources(
                game_id, [bayesian_demand], {"bayesian_reasoning": True}
            )

            return allocation

        except Exception as e:
            logger.error(f"Error requesting attention allocation: {e}")
            return None