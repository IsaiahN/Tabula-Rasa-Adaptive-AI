#!/usr/bin/env python3
"""
Interleaved Training Enhancer for Tabula Rasa

Enhances existing EWC system with interleaved training to prevent
catastrophic forgetting, building on existing meta-learning systems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import math
import random
from collections import deque
import json

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks for interleaved training."""
    ARC_PUZZLE = "arc_puzzle"
    PATTERN_RECOGNITION = "pattern_recognition"
    SPATIAL_REASONING = "spatial_reasoning"
    LOGICAL_REASONING = "logical_reasoning"
    MEMORY_TASK = "memory_task"
    NOVEL_TASK = "novel_task"

class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class TrainingMode(Enum):
    """Training modes for interleaved learning."""
    INTERLEAVED = "interleaved"
    BLOCKED = "blocked"
    RANDOM = "random"
    CURRICULUM = "curriculum"

@dataclass
class TrainingTask:
    """Represents a training task."""
    task_id: str
    task_type: TaskType
    difficulty: DifficultyLevel
    input_data: Dict[str, Any]
    target_output: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class InterleavedSchedule:
    """Schedule for interleaved training."""
    tasks: List[TrainingTask]
    mode: TrainingMode
    difficulty_progression: List[DifficultyLevel]
    interleaving_ratio: float
    total_epochs: int
    current_epoch: int = 0

@dataclass
class RehearsalBuffer:
    """Buffer for storing old tasks for rehearsal."""
    max_size: int
    tasks: deque
    importance_scores: Dict[str, float]
    access_counts: Dict[str, int]

class CurriculumScheduler:
    """Scheduler for curriculum learning with interleaved training."""
    
    def __init__(self, 
                 difficulty_levels: List[DifficultyLevel] = None,
                 interleaving_ratio: float = 0.3,
                 progression_rate: float = 0.1):
        self.difficulty_levels = difficulty_levels or list(DifficultyLevel)
        self.interleaving_ratio = interleaving_ratio
        self.progression_rate = progression_rate
        
        # Current curriculum state
        self.current_difficulty_idx = 0
        self.task_performance_history = {}
        self.difficulty_performance = {level: [] for level in self.difficulty_levels}
        
        logger.info("Curriculum Scheduler initialized")
    
    def create_interleaved_curriculum(self, 
                                    tasks: List[TrainingTask], 
                                    mode: TrainingMode = TrainingMode.INTERLEAVED) -> InterleavedSchedule:
        """Create an interleaved training curriculum."""
        
        if mode == TrainingMode.INTERLEAVED:
            scheduled_tasks = self._create_interleaved_schedule(tasks)
        elif mode == TrainingMode.CURRICULUM:
            scheduled_tasks = self._create_curriculum_schedule(tasks)
        elif mode == TrainingMode.RANDOM:
            scheduled_tasks = self._create_random_schedule(tasks)
        else:  # BLOCKED
            scheduled_tasks = self._create_blocked_schedule(tasks)
        
        # Calculate total epochs
        total_epochs = len(scheduled_tasks) // 10  # Assuming 10 tasks per epoch
        
        schedule = InterleavedSchedule(
            tasks=scheduled_tasks,
            mode=mode,
            difficulty_progression=self.difficulty_levels,
            interleaving_ratio=self.interleaving_ratio,
            total_epochs=total_epochs
        )
        
        return schedule
    
    def _create_interleaved_schedule(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """Create interleaved schedule mixing different task types."""
        # Group tasks by type
        tasks_by_type = {}
        for task in tasks:
            task_type = task.task_type
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(task)
        
        # Create interleaved schedule
        interleaved_tasks = []
        max_tasks_per_type = max(len(task_list) for task_list in tasks_by_type.values())
        
        for i in range(max_tasks_per_type):
            for task_type, task_list in tasks_by_type.items():
                if i < len(task_list):
                    interleaved_tasks.append(task_list[i])
        
        return interleaved_tasks
    
    def _create_curriculum_schedule(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """Create curriculum schedule with difficulty progression."""
        # Group tasks by difficulty
        tasks_by_difficulty = {}
        for task in tasks:
            difficulty = task.difficulty
            if difficulty not in tasks_by_difficulty:
                tasks_by_difficulty[difficulty] = []
            tasks_by_difficulty[difficulty].append(task)
        
        # Create curriculum schedule
        curriculum_tasks = []
        
        for difficulty in self.difficulty_levels:
            if difficulty in tasks_by_difficulty:
                # Add tasks from current difficulty level
                curriculum_tasks.extend(tasks_by_difficulty[difficulty])
                
                # Add some interleaving with previous levels
                if self.current_difficulty_idx > 0:
                    prev_difficulty = self.difficulty_levels[self.current_difficulty_idx - 1]
                    if prev_difficulty in tasks_by_difficulty:
                        # Add 30% of previous difficulty tasks
                        prev_tasks = tasks_by_difficulty[prev_difficulty]
                        num_prev_tasks = int(len(prev_tasks) * self.interleaving_ratio)
                        curriculum_tasks.extend(prev_tasks[:num_prev_tasks])
        
        return curriculum_tasks
    
    def _create_random_schedule(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """Create random schedule."""
        random_tasks = tasks.copy()
        random.shuffle(random_tasks)
        return random_tasks
    
    def _create_blocked_schedule(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """Create blocked schedule (traditional training)."""
        # Group by task type and train each type in blocks
        blocked_tasks = []
        tasks_by_type = {}
        
        for task in tasks:
            task_type = task.task_type
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(task)
        
        # Add all tasks of each type in sequence
        for task_type, task_list in tasks_by_type.items():
            blocked_tasks.extend(task_list)
        
        return blocked_tasks
    
    def update_curriculum_progress(self, task_performance: Dict[str, float]):
        """Update curriculum based on task performance."""
        # Update performance history
        self.task_performance_history.update(task_performance)
        
        # Update difficulty performance
        for task_id, performance in task_performance.items():
            # Find difficulty level for this task (simplified)
            difficulty = DifficultyLevel.MEDIUM  # Placeholder
            self.difficulty_performance[difficulty].append(performance)
        
        # Check if we should progress to next difficulty
        if self._should_progress_difficulty():
            self.current_difficulty_idx = min(
                self.current_difficulty_idx + 1, 
                len(self.difficulty_levels) - 1
            )
            logger.info(f"Progressed to difficulty level: {self.difficulty_levels[self.current_difficulty_idx].value}")
    
    def _should_progress_difficulty(self) -> bool:
        """Check if we should progress to next difficulty level."""
        if self.current_difficulty_idx >= len(self.difficulty_levels) - 1:
            return False
        
        current_difficulty = self.difficulty_levels[self.current_difficulty_idx]
        performance_history = self.difficulty_performance.get(current_difficulty, [])
        
        if len(performance_history) < 10:  # Need minimum samples
            return False
        
        # Check if performance is consistently good
        recent_performance = performance_history[-10:]
        avg_performance = np.mean(recent_performance)
        
        return avg_performance > 0.8  # 80% success rate threshold

class GenerativeReplay:
    """Generative replay for creating synthetic training examples."""
    
    def __init__(self, 
                 model_capacity: int = 1000,
                 generation_ratio: float = 0.2):
        self.model_capacity = model_capacity
        self.generation_ratio = generation_ratio
        self.generated_examples = deque(maxlen=model_capacity)
        self.generation_model = None  # Placeholder for actual generative model
        
        logger.info("Generative Replay initialized")
    
    def generate_examples(self, 
                         original_tasks: List[TrainingTask], 
                         num_examples: int) -> List[TrainingTask]:
        """Generate synthetic training examples."""
        if not original_tasks:
            return []
        
        # Calculate number of examples to generate
        num_to_generate = min(num_examples, int(len(original_tasks) * self.generation_ratio))
        
        generated_tasks = []
        
        for _ in range(num_to_generate):
            # Select a random original task as template
            template_task = random.choice(original_tasks)
            
            # Generate synthetic task based on template
            synthetic_task = self._generate_synthetic_task(template_task)
            
            if synthetic_task:
                generated_tasks.append(synthetic_task)
                self.generated_examples.append(synthetic_task)
        
        return generated_tasks
    
    def _generate_synthetic_task(self, template_task: TrainingTask) -> Optional[TrainingTask]:
        """Generate a synthetic task based on template."""
        try:
            # Create synthetic task with modified input data
            synthetic_input = self._modify_input_data(template_task.input_data)
            synthetic_output = self._modify_output_data(template_task.target_output)
            
            synthetic_task = TrainingTask(
                task_id=f"synthetic_{template_task.task_id}_{int(time.time())}",
                task_type=template_task.task_type,
                difficulty=template_task.difficulty,
                input_data=synthetic_input,
                target_output=synthetic_output,
                metadata={
                    'generated': True,
                    'template_task_id': template_task.task_id,
                    'generation_method': 'template_based'
                }
            )
            
            return synthetic_task
            
        except Exception as e:
            logger.warning(f"Failed to generate synthetic task: {e}")
            return None
    
    def _modify_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify input data to create synthetic example."""
        synthetic_input = input_data.copy()
        
        # Add noise or variations
        for key, value in synthetic_input.items():
            if isinstance(value, (int, float)):
                # Add small random variation
                noise = np.random.normal(0, 0.1)
                synthetic_input[key] = value + noise
            elif isinstance(value, list) and len(value) > 0:
                # Shuffle list or add noise to numeric elements
                if isinstance(value[0], (int, float)):
                    synthetic_input[key] = [v + np.random.normal(0, 0.1) for v in value]
                else:
                    synthetic_input[key] = value.copy()
                    random.shuffle(synthetic_input[key])
        
        return synthetic_input
    
    def _modify_output_data(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify output data to create synthetic example."""
        synthetic_output = output_data.copy()
        
        # Add small variations to output
        for key, value in synthetic_output.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, 0.05)
                synthetic_output[key] = value + noise
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    synthetic_output[key] = [v + np.random.normal(0, 0.05) for v in value]
        
        return synthetic_output

class InterleavedTrainingEnhancer:
    """Main enhancer for interleaved training to prevent catastrophic forgetting."""
    
    def __init__(self, 
                 rehearsal_buffer_size: int = 1000,
                 interleaving_ratio: float = 0.3,
                 curriculum_enabled: bool = True,
                 generative_replay_enabled: bool = True):
        self.rehearsal_buffer_size = rehearsal_buffer_size
        self.interleaving_ratio = interleaving_ratio
        self.curriculum_enabled = curriculum_enabled
        self.generative_replay_enabled = generative_replay_enabled
        
        # Initialize components
        self.curriculum_scheduler = CurriculumScheduler(
            interleaving_ratio=interleaving_ratio
        )
        self.rehearsal_buffer = RehearsalBuffer(
            max_size=rehearsal_buffer_size,
            tasks=deque(maxlen=rehearsal_buffer_size),
            importance_scores={},
            access_counts={}
        )
        self.generative_replay = GenerativeReplay() if generative_replay_enabled else None
        
        # Training statistics
        self.training_stats = {
            'total_tasks_processed': 0,
            'interleaved_tasks': 0,
            'rehearsal_tasks': 0,
            'generated_tasks': 0,
            'curriculum_progressions': 0,
            'forgetting_events': 0
        }
        
        logger.info("Interleaved Training Enhancer initialized")
    
    def create_enhanced_training_schedule(self, 
                                        tasks: List[TrainingTask],
                                        mode: TrainingMode = TrainingMode.INTERLEAVED) -> InterleavedSchedule:
        """Create enhanced training schedule with interleaving and rehearsal."""
        
        # Create base schedule
        base_schedule = self.curriculum_scheduler.create_interleaved_curriculum(tasks, mode)
        
        # Add rehearsal tasks
        rehearsal_tasks = self._get_rehearsal_tasks()
        base_schedule.tasks.extend(rehearsal_tasks)
        
        # Add generated tasks if enabled
        if self.generative_replay_enabled and self.generative_replay:
            generated_tasks = self.generative_replay.generate_examples(tasks, len(tasks) // 5)
            base_schedule.tasks.extend(generated_tasks)
            self.training_stats['generated_tasks'] += len(generated_tasks)
        
        # Shuffle final schedule to ensure proper interleaving
        random.shuffle(base_schedule.tasks)
        
        # Update statistics
        self.training_stats['total_tasks_processed'] += len(base_schedule.tasks)
        self.training_stats['interleaved_tasks'] += len(tasks)
        self.training_stats['rehearsal_tasks'] += len(rehearsal_tasks)
        
        return base_schedule
    
    def _get_rehearsal_tasks(self) -> List[TrainingTask]:
        """Get tasks from rehearsal buffer."""
        rehearsal_tasks = []
        
        if not self.rehearsal_buffer.tasks:
            return rehearsal_tasks
        
        # Select tasks based on importance and access frequency
        num_rehearsal_tasks = int(len(self.rehearsal_buffer.tasks) * self.interleaving_ratio)
        
        # Sort tasks by importance score
        sorted_tasks = sorted(
            self.rehearsal_buffer.tasks,
            key=lambda task: self.rehearsal_buffer.importance_scores.get(task.task_id, 0.0),
            reverse=True
        )
        
        # Select top tasks for rehearsal
        rehearsal_tasks = sorted_tasks[:num_rehearsal_tasks]
        
        # Update access counts
        for task in rehearsal_tasks:
            task_id = task.task_id
            self.rehearsal_buffer.access_counts[task_id] = self.rehearsal_buffer.access_counts.get(task_id, 0) + 1
        
        return rehearsal_tasks
    
    def add_task_to_rehearsal_buffer(self, task: TrainingTask, importance_score: float = 1.0):
        """Add task to rehearsal buffer."""
        self.rehearsal_buffer.tasks.append(task)
        self.rehearsal_buffer.importance_scores[task.task_id] = importance_score
        self.rehearsal_buffer.access_counts[task.task_id] = 0
    
    def update_task_importance(self, task_id: str, new_importance: float):
        """Update importance score for a task."""
        if task_id in self.rehearsal_buffer.importance_scores:
            self.rehearsal_buffer.importance_scores[task_id] = new_importance
    
    def detect_catastrophic_forgetting(self, 
                                     old_task_performance: Dict[str, float],
                                     new_task_performance: Dict[str, float]) -> bool:
        """Detect if catastrophic forgetting has occurred."""
        if not old_task_performance or not new_task_performance:
            return False
        
        # Calculate performance drop
        common_tasks = set(old_task_performance.keys()) & set(new_task_performance.keys())
        
        if not common_tasks:
            return False
        
        performance_drops = []
        for task_id in common_tasks:
            old_perf = old_task_performance[task_id]
            new_perf = new_task_performance[task_id]
            drop = old_perf - new_perf
            performance_drops.append(drop)
        
        # Check if average performance drop exceeds threshold
        avg_drop = np.mean(performance_drops)
        forgetting_threshold = 0.2  # 20% performance drop
        
        if avg_drop > forgetting_threshold:
            self.training_stats['forgetting_events'] += 1
            return True
        
        return False
    
    def update_curriculum_progress(self, task_performance: Dict[str, float]):
        """Update curriculum progress based on task performance."""
        if self.curriculum_enabled:
            self.curriculum_scheduler.update_curriculum_progress(task_performance)
            
            # Check if curriculum progressed
            if self.curriculum_scheduler.current_difficulty_idx > 0:
                self.training_stats['curriculum_progressions'] += 1
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about interleaved training."""
        return {
            'total_tasks_processed': self.training_stats['total_tasks_processed'],
            'interleaved_tasks': self.training_stats['interleaved_tasks'],
            'rehearsal_tasks': self.training_stats['rehearsal_tasks'],
            'generated_tasks': self.training_stats['generated_tasks'],
            'curriculum_progressions': self.training_stats['curriculum_progressions'],
            'forgetting_events': self.training_stats['forgetting_events'],
            'rehearsal_buffer_size': len(self.rehearsal_buffer.tasks),
            'current_difficulty_level': self.curriculum_scheduler.difficulty_levels[
                self.curriculum_scheduler.current_difficulty_idx
            ].value if self.curriculum_enabled else 'N/A',
            'interleaving_ratio': self.interleaving_ratio,
            'curriculum_enabled': self.curriculum_enabled,
            'generative_replay_enabled': self.generative_replay_enabled
        }

# Factory function for easy integration
def create_interleaved_training_enhancer(**kwargs) -> InterleavedTrainingEnhancer:
    """Create a configured interleaved training enhancer."""
    return InterleavedTrainingEnhancer(**kwargs)
