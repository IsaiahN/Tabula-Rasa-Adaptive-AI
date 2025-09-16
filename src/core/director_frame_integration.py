"""
Director Frame Integration System
Integrates frame dynamics analysis with the Director's meta-cognitive capabilities.
Generates strategic inferences and communicates them to Architect and Governor.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import json
import time
from datetime import datetime

from .frame_dynamics_analyzer import (
    FrameDynamicsAnalyzer, 
    PhysicsInference, 
    GameMechanicInference,
    PhysicsType,
    GameMechanic
)

class DirectorFrameIntegration:
    """
    Director's frame analysis integration system.
    Uses frame dynamics analysis to generate strategic inferences for the Architect and Governor.
    """
    
    def __init__(self, max_frame_history: int = 50):
        self.frame_analyzer = FrameDynamicsAnalyzer()
        self.frame_history: Dict[str, deque] = {}  # game_id -> frame history
        self.max_frame_history = max_frame_history
        
        # Strategic inference storage
        self.strategic_inferences: Dict[str, List[Dict]] = {}  # game_id -> inferences
        self.physics_models: Dict[str, Dict] = {}  # game_id -> physics model
        self.mechanic_models: Dict[str, Dict] = {}  # game_id -> mechanic model
        
        # Communication channels
        self.architect_guidance: Dict[str, List[Dict]] = {}  # game_id -> guidance
        self.governor_guidance: Dict[str, List[Dict]] = {}  # game_id -> guidance
        
    def process_frame_sequence(self, 
                             frames: List[np.ndarray], 
                             game_id: str,
                             action_history: List[Tuple[int, Tuple[int, int]]]) -> Dict[str, Any]:
        """
        Process a sequence of frames to generate strategic inferences.
        
        Args:
            frames: List of frame arrays
            game_id: Game identifier
            action_history: History of actions taken
            
        Returns:
            Dictionary containing strategic inferences and guidance
        """
        # Initialize frame history for this game
        if game_id not in self.frame_history:
            self.frame_history[game_id] = deque(maxlen=self.max_frame_history)
        
        # Add frames to history
        timestamps = [time.time()] * len(frames)
        for frame in frames:
            self.frame_history[game_id].append(frame)
        
        # Analyze frame sequence for dynamics
        analysis_result = self.frame_analyzer.analyze_frame_sequence(
            frames, timestamps, game_id, action_history
        )
        
        # Generate strategic inferences
        strategic_inferences = self._generate_strategic_inferences(analysis_result, game_id)
        
        # Update physics and mechanic models
        self._update_physics_model(analysis_result, game_id)
        self._update_mechanic_model(analysis_result, game_id)
        
        # Generate guidance for Architect and Governor
        architect_guidance = self._generate_architect_guidance(analysis_result, game_id)
        governor_guidance = self._generate_governor_guidance(analysis_result, game_id)
        
        # Store guidance
        self.architect_guidance[game_id] = architect_guidance
        self.governor_guidance[game_id] = governor_guidance
        
        return {
            "strategic_inferences": strategic_inferences,
            "architect_guidance": architect_guidance,
            "governor_guidance": governor_guidance,
            "physics_model": self.physics_models.get(game_id, {}),
            "mechanic_model": self.mechanic_models.get(game_id, {}),
            "confidence_score": analysis_result.get("confidence_score", 0.0)
        }
    
    def _generate_strategic_inferences(self, analysis_result: Dict[str, Any], game_id: str) -> List[Dict]:
        """Generate strategic inferences from frame analysis."""
        inferences = []
        
        # Physics-based strategic inferences
        for physics_inf in analysis_result.get("physics_inferences", []):
            inference = self._create_physics_strategic_inference(physics_inf, game_id)
            if inference:
                inferences.append(inference)
        
        # Mechanic-based strategic inferences
        for mechanic_inf in analysis_result.get("mechanic_inferences", []):
            inference = self._create_mechanic_strategic_inference(mechanic_inf, game_id)
            if inference:
                inferences.append(inference)
        
        # Store inferences
        if game_id not in self.strategic_inferences:
            self.strategic_inferences[game_id] = []
        self.strategic_inferences[game_id].extend(inferences)
        
        return inferences
    
    def _create_physics_strategic_inference(self, physics_inf: Dict, game_id: str) -> Optional[Dict]:
        """Create strategic inference from physics analysis."""
        physics_type = physics_inf.get("physics_type", "")
        confidence = physics_inf.get("confidence", 0.0)
        
        if confidence < 0.5:  # Only use high-confidence inferences
            return None
        
        # Generate strategic implications based on physics type
        strategic_implications = self._get_physics_strategic_implications(physics_type)
        
        return {
            "type": "physics_strategic_inference",
            "game_id": game_id,
            "physics_type": physics_type,
            "confidence": confidence,
            "strategic_implications": strategic_implications,
            "evidence": physics_inf.get("evidence", []),
            "spatial_region": physics_inf.get("spatial_region", (0, 0, 64, 64)),
            "temporal_span": physics_inf.get("temporal_span", 1),
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_mechanic_strategic_inference(self, mechanic_inf: Dict, game_id: str) -> Optional[Dict]:
        """Create strategic inference from mechanic analysis."""
        mechanic_type = mechanic_inf.get("mechanic", "")
        confidence = mechanic_inf.get("confidence", 0.0)
        
        if confidence < 0.5:  # Only use high-confidence inferences
            return None
        
        # Generate strategic implications based on mechanic type
        strategic_implications = self._get_mechanic_strategic_implications(mechanic_type)
        
        return {
            "type": "mechanic_strategic_inference",
            "game_id": game_id,
            "mechanic_type": mechanic_type,
            "confidence": confidence,
            "strategic_implications": strategic_implications,
            "evidence": mechanic_inf.get("evidence", []),
            "required_actions": mechanic_inf.get("required_actions", []),
            "success_criteria": mechanic_inf.get("success_criteria", []),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_physics_strategic_implications(self, physics_type: str) -> List[str]:
        """Get strategic implications for physics types."""
        implications_map = {
            "gravity": [
                "Use gravity to manipulate objects downward",
                "Create gravity-based object arrangements",
                "Leverage falling objects for puzzle solutions"
            ],
            "momentum": [
                "Use momentum to predict object movement",
                "Create momentum-based object interactions",
                "Leverage momentum conservation for solutions"
            ],
            "collision": [
                "Use collisions to redirect object movement",
                "Create collision-based object interactions",
                "Leverage collision effects for puzzle solutions"
            ],
            "color_transition": [
                "Use color changes to track object states",
                "Create color-based conditional logic",
                "Leverage color transitions for state management"
            ],
            "shape_morphing": [
                "Use shape changes to track object functions",
                "Create shape-based object interactions",
                "Leverage shape morphing for puzzle solutions"
            ]
        }
        
        return implications_map.get(physics_type, ["Unknown physics type"])
    
    def _get_mechanic_strategic_implications(self, mechanic_type: str) -> List[str]:
        """Get strategic implications for mechanic types."""
        implications_map = {
            "puzzle_solving": [
                "Focus on systematic progression toward goal state",
                "Identify intermediate steps and sub-goals",
                "Use logical reasoning to solve puzzles"
            ],
            "pattern_matching": [
                "Identify and match repeating patterns",
                "Use pattern recognition for object classification",
                "Leverage patterns for predictive behavior"
            ],
            "object_manipulation": [
                "Use actions to manipulate objects toward goal",
                "Identify which objects respond to which actions",
                "Create sequences of object manipulations"
            ],
            "spatial_reasoning": [
                "Use spatial relationships for object placement",
                "Identify spatial constraints and requirements",
                "Leverage spatial logic for puzzle solutions"
            ],
            "temporal_sequencing": [
                "Use temporal order for action sequences",
                "Identify time-dependent object behaviors",
                "Leverage timing for puzzle solutions"
            ]
        }
        
        return implications_map.get(mechanic_type, ["Unknown mechanic type"])
    
    def _update_physics_model(self, analysis_result: Dict[str, Any], game_id: str):
        """Update physics model for the game."""
        if game_id not in self.physics_models:
            self.physics_models[game_id] = {
                "detected_physics": [],
                "confidence_scores": [],
                "spatial_regions": [],
                "temporal_patterns": []
            }
        
        # Add new physics detections
        for physics_inf in analysis_result.get("physics_inferences", []):
            self.physics_models[game_id]["detected_physics"].append(physics_inf.get("physics_type", ""))
            self.physics_models[game_id]["confidence_scores"].append(physics_inf.get("confidence", 0.0))
            self.physics_models[game_id]["spatial_regions"].append(physics_inf.get("spatial_region", (0, 0, 64, 64)))
            self.physics_models[game_id]["temporal_patterns"].append(physics_inf.get("temporal_span", 1))
    
    def _update_mechanic_model(self, analysis_result: Dict[str, Any], game_id: str):
        """Update mechanic model for the game."""
        if game_id not in self.mechanic_models:
            self.mechanic_models[game_id] = {
                "detected_mechanics": [],
                "confidence_scores": [],
                "required_actions": [],
                "success_criteria": []
            }
        
        # Add new mechanic detections
        for mechanic_inf in analysis_result.get("mechanic_inferences", []):
            self.mechanic_models[game_id]["detected_mechanics"].append(mechanic_inf.get("mechanic", ""))
            self.mechanic_models[game_id]["confidence_scores"].append(mechanic_inf.get("confidence", 0.0))
            self.mechanic_models[game_id]["required_actions"].extend(mechanic_inf.get("required_actions", []))
            self.mechanic_models[game_id]["success_criteria"].extend(mechanic_inf.get("success_criteria", []))
    
    def _generate_architect_guidance(self, analysis_result: Dict[str, Any], game_id: str) -> List[Dict]:
        """Generate guidance for the Architect system."""
        guidance = []
        
        # Physics-based guidance for Architect
        for physics_inf in analysis_result.get("physics_inferences", []):
            if physics_inf.get("confidence", 0.0) > 0.6:
                guidance.append({
                    "type": "physics_architecture",
                    "physics_type": physics_inf.get("physics_type", ""),
                    "confidence": physics_inf.get("confidence", 0.0),
                    "implication": physics_inf.get("strategic_implication", ""),
                    "spatial_region": physics_inf.get("spatial_region", (0, 0, 64, 64)),
                    "priority": "high" if physics_inf.get("confidence", 0.0) > 0.8 else "medium"
                })
        
        # Mechanic-based guidance for Architect
        for mechanic_inf in analysis_result.get("mechanic_inferences", []):
            if mechanic_inf.get("confidence", 0.0) > 0.6:
                guidance.append({
                    "type": "mechanic_architecture",
                    "mechanic_type": mechanic_inf.get("mechanic", ""),
                    "confidence": mechanic_inf.get("confidence", 0.0),
                    "guidance": mechanic_inf.get("strategic_guidance", ""),
                    "required_actions": mechanic_inf.get("required_actions", []),
                    "priority": "high" if mechanic_inf.get("confidence", 0.0) > 0.8 else "medium"
                })
        
        return guidance
    
    def _generate_governor_guidance(self, analysis_result: Dict[str, Any], game_id: str) -> List[Dict]:
        """Generate guidance for the Governor system."""
        guidance = []
        
        # Action prioritization guidance
        action_priorities = analysis_result.get("strategic_guidance", {}).get("action_priorities", [])
        if action_priorities:
            guidance.append({
                "type": "action_prioritization",
                "priorities": action_priorities,
                "reasoning": "Based on detected game mechanics",
                "priority": "high"
            })
        
        # Resource allocation guidance
        confidence_score = analysis_result.get("confidence_score", 0.0)
        if confidence_score > 0.7:
            guidance.append({
                "type": "resource_allocation",
                "recommendation": "increase_analysis_resources",
                "reasoning": f"High confidence analysis ({confidence_score:.2f})",
                "priority": "medium"
            })
        
        # Success strategy guidance
        success_strategies = analysis_result.get("strategic_guidance", {}).get("success_strategies", [])
        if success_strategies:
            guidance.append({
                "type": "success_strategy",
                "strategies": success_strategies,
                "reasoning": "Based on frame dynamics analysis",
                "priority": "high"
            })
        
        return guidance
    
    def get_architect_guidance(self, game_id: str) -> List[Dict]:
        """Get current guidance for Architect system."""
        return self.architect_guidance.get(game_id, [])
    
    def get_governor_guidance(self, game_id: str) -> List[Dict]:
        """Get current guidance for Governor system."""
        return self.governor_guidance.get(game_id, [])
    
    def get_strategic_inferences(self, game_id: str) -> List[Dict]:
        """Get current strategic inferences for a game."""
        return self.strategic_inferences.get(game_id, [])
    
    def get_physics_model(self, game_id: str) -> Dict:
        """Get current physics model for a game."""
        return self.physics_models.get(game_id, {})
    
    def get_mechanic_model(self, game_id: str) -> Dict:
        """Get current mechanic model for a game."""
        return self.mechanic_models.get(game_id, {})
    
    def clear_game_data(self, game_id: str):
        """Clear all data for a specific game."""
        if game_id in self.frame_history:
            del self.frame_history[game_id]
        if game_id in self.strategic_inferences:
            del self.strategic_inferences[game_id]
        if game_id in self.physics_models:
            del self.physics_models[game_id]
        if game_id in self.mechanic_models:
            del self.mechanic_models[game_id]
        if game_id in self.architect_guidance:
            del self.architect_guidance[game_id]
        if game_id in self.governor_guidance:
            del self.governor_guidance[game_id]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "active_games": list(self.frame_history.keys()),
            "total_strategic_inferences": sum(len(inferences) for inferences in self.strategic_inferences.values()),
            "physics_models_count": len(self.physics_models),
            "mechanic_models_count": len(self.mechanic_models),
            "architect_guidance_count": sum(len(guidance) for guidance in self.architect_guidance.values()),
            "governor_guidance_count": sum(len(guidance) for guidance in self.governor_guidance.values())
        }
