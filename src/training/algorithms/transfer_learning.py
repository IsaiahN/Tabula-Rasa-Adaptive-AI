"""
Transfer Learning Algorithm

Implements advanced transfer learning techniques for knowledge transfer
between related tasks using the modular architecture.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..interfaces import ComponentInterface, LearningInterface
from ..caching import CacheManager, CacheConfig
from ..monitoring import TrainingMonitor


class TransferStrategy(Enum):
    """Transfer learning strategies."""
    FEATURE_EXTRACTION = "feature_extraction"
    FINE_TUNING = "fine_tuning"
    DOMAIN_ADAPTATION = "domain_adaptation"
    MULTI_TASK = "multi_task"
    PROGRESSIVE = "progressive"


@dataclass
class TransferConfig:
    """Configuration for transfer learning algorithm."""
    source_domain: str = "general"
    target_domain: str = "specific"
    transfer_strategy: TransferStrategy = TransferStrategy.FINE_TUNING
    freeze_layers: int = 0
    learning_rate_multiplier: float = 0.1
    adaptation_epochs: int = 50
    knowledge_retention: float = 0.8
    similarity_threshold: float = 0.7
    transfer_confidence_threshold: float = 0.6


class TransferLearningAlgorithm(ComponentInterface, LearningInterface):
    """
    Advanced transfer learning algorithm that intelligently transfers
    knowledge between related tasks using the modular architecture.
    """
    
    def __init__(self, config: TransferConfig, cache_config: Optional[CacheConfig] = None):
        """Initialize the transfer learning algorithm."""
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.training_monitor = TrainingMonitor("transfer_learning")
        
        # Transfer learning state
        self.source_models: Dict[str, Dict[str, Any]] = {}
        self.target_models: Dict[str, Dict[str, Any]] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        self.domain_similarities: Dict[Tuple[str, str], float] = {}
        
        # Knowledge base
        self.knowledge_base: Dict[str, List[Dict[str, Any]]] = {}
        self.feature_mappings: Dict[Tuple[str, str], np.ndarray] = {}
        self.transfer_weights: Dict[str, float] = {}
        
        # Performance tracking
        self.transfer_success_rate: float = 0.0
        self.average_transfer_gain: float = 0.0
        self.knowledge_retention_rate: float = 0.0
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the transfer learning algorithm."""
        try:
            self.cache.initialize()
            self.training_monitor.start_monitoring()
            self._initialized = True
            self.logger.info("Transfer learning algorithm initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize transfer learning algorithm: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'TransferLearningAlgorithm',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'source_models': len(self.source_models),
                'target_models': len(self.target_models),
                'transfer_success_rate': self.transfer_success_rate,
                'average_transfer_gain': self.average_transfer_gain,
                'knowledge_retention_rate': self.knowledge_retention_rate
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.training_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Transfer learning algorithm cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from an experience and update knowledge base."""
        try:
            source_domain = experience.get('source_domain', 'unknown')
            target_domain = experience.get('target_domain', 'unknown')
            transfer_performance = experience.get('transfer_performance', 0.0)
            knowledge_type = experience.get('knowledge_type', 'general')
            
            # Store in knowledge base
            if source_domain not in self.knowledge_base:
                self.knowledge_base[source_domain] = []
            
            knowledge_entry = {
                'experience': experience,
                'performance': transfer_performance,
                'timestamp': datetime.now(),
                'knowledge_type': knowledge_type
            }
            
            self.knowledge_base[source_domain].append(knowledge_entry)
            
            # Update domain similarity
            similarity = self._calculate_domain_similarity(source_domain, target_domain)
            self.domain_similarities[(source_domain, target_domain)] = similarity
            
            # Update transfer weights
            self._update_transfer_weights(source_domain, target_domain, transfer_performance)
            
            # Record transfer history
            self.transfer_history.append({
                'source_domain': source_domain,
                'target_domain': target_domain,
                'performance': transfer_performance,
                'similarity': similarity,
                'timestamp': datetime.now()
            })
            
            # Cache the experience
            cache_key = f"transfer_experience_{len(self.transfer_history)}"
            self.cache.set(cache_key, experience, ttl=3600)
            
            self.logger.debug(f"Learned from transfer experience: {source_domain} -> {target_domain}")
            
        except Exception as e:
            self.logger.error(f"Error learning from experience: {e}")
    
    def apply_learning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transfer learning to a new context."""
        try:
            source_domain = context.get('source_domain', self.config.source_domain)
            target_domain = context.get('target_domain', self.config.target_domain)
            target_task = context.get('target_task', {})
            
            # Find best source model for transfer
            best_source = self._find_best_source_model(source_domain, target_domain)
            
            if not best_source:
                return {'error': 'No suitable source model found'}
            
            # Calculate transfer strategy
            transfer_strategy = self._select_transfer_strategy(source_domain, target_domain)
            
            # Perform knowledge transfer
            transfer_result = self._perform_transfer(
                best_source,
                target_task,
                transfer_strategy
            )
            
            # Calculate transfer confidence
            transfer_confidence = self._calculate_transfer_confidence(
                source_domain, target_domain, transfer_result
            )
            
            result = {
                'source_model': best_source,
                'transfer_strategy': transfer_strategy.value,
                'transferred_knowledge': transfer_result,
                'transfer_confidence': transfer_confidence,
                'domain_similarity': self.domain_similarities.get((source_domain, target_domain), 0.0),
                'estimated_performance_gain': self._estimate_performance_gain(transfer_result)
            }
            
            # Cache the transfer result
            cache_key = f"transfer_{source_domain}_{target_domain}_{datetime.now().timestamp()}"
            self.cache.set(cache_key, result, ttl=1800)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying transfer learning: {e}")
            return {'error': str(e)}
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get transfer learning statistics."""
        try:
            # Calculate transfer success rate
            successful_transfers = [
                t for t in self.transfer_history 
                if t['performance'] > self.config.transfer_confidence_threshold
            ]
            
            self.transfer_success_rate = (
                len(successful_transfers) / max(1, len(self.transfer_history))
            )
            
            # Calculate average transfer gain
            if self.transfer_history:
                performances = [t['performance'] for t in self.transfer_history]
                self.average_transfer_gain = np.mean(performances)
            
            # Calculate knowledge retention rate
            self.knowledge_retention_rate = self._calculate_knowledge_retention()
            
            # Domain-specific statistics
            domain_stats = {}
            for domain, knowledge in self.knowledge_base.items():
                if knowledge:
                    performances = [k['performance'] for k in knowledge]
                    domain_stats[domain] = {
                        'knowledge_count': len(knowledge),
                        'average_performance': np.mean(performances),
                        'best_performance': max(performances),
                        'knowledge_types': list(set(k['knowledge_type'] for k in knowledge))
                    }
            
            # Transfer pair statistics
            transfer_pairs = {}
            for (source, target), similarity in self.domain_similarities.items():
                pair_key = f"{source}->{target}"
                pair_transfers = [
                    t for t in self.transfer_history 
                    if t['source_domain'] == source and t['target_domain'] == target
                ]
                
                if pair_transfers:
                    transfer_pairs[pair_key] = {
                        'similarity': similarity,
                        'transfer_count': len(pair_transfers),
                        'average_performance': np.mean([t['performance'] for t in pair_transfers]),
                        'success_rate': len([t for t in pair_transfers if t['performance'] > 0.7]) / len(pair_transfers)
                    }
            
            return {
                'transfer_success_rate': self.transfer_success_rate,
                'average_transfer_gain': self.average_transfer_gain,
                'knowledge_retention_rate': self.knowledge_retention_rate,
                'total_transfers': len(self.transfer_history),
                'domain_statistics': domain_stats,
                'transfer_pairs': transfer_pairs,
                'source_models': len(self.source_models),
                'target_models': len(self.target_models)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning stats: {e}")
            return {'error': str(e)}
    
    def register_source_model(self, model_id: str, model_data: Dict[str, Any], 
                            domain: str) -> bool:
        """Register a source model for transfer learning."""
        try:
            self.source_models[model_id] = {
                'model_data': model_data,
                'domain': domain,
                'registered_at': datetime.now(),
                'transfer_count': 0
            }
            
            # Cache the model
            self.cache.set(f"source_model_{model_id}", model_data, ttl=86400)
            
            self.logger.info(f"Registered source model {model_id} for domain {domain}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering source model {model_id}: {e}")
            return False
    
    def register_target_model(self, model_id: str, model_data: Dict[str, Any], 
                            domain: str) -> bool:
        """Register a target model for transfer learning."""
        try:
            self.target_models[model_id] = {
                'model_data': model_data,
                'domain': domain,
                'registered_at': datetime.now(),
                'transfer_count': 0
            }
            
            # Cache the model
            self.cache.set(f"target_model_{model_id}", model_data, ttl=86400)
            
            self.logger.info(f"Registered target model {model_id} for domain {domain}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering target model {model_id}: {e}")
            return False
    
    def _find_best_source_model(self, source_domain: str, target_domain: str) -> Optional[str]:
        """Find the best source model for transfer."""
        try:
            # Filter models by domain
            domain_models = {
                model_id: model for model_id, model in self.source_models.items()
                if model['domain'] == source_domain
            }
            
            if not domain_models:
                return None
            
            # Calculate transfer scores
            transfer_scores = {}
            for model_id, model in domain_models.items():
                score = self._calculate_transfer_score(model, target_domain)
                transfer_scores[model_id] = score
            
            # Return model with highest score
            best_model = max(transfer_scores.items(), key=lambda x: x[1])
            return best_model[0] if best_model[1] > 0.5 else None
            
        except Exception as e:
            self.logger.error(f"Error finding best source model: {e}")
            return None
    
    def _select_transfer_strategy(self, source_domain: str, target_domain: str) -> TransferStrategy:
        """Select the best transfer strategy for the domain pair."""
        similarity = self.domain_similarities.get((source_domain, target_domain), 0.0)
        
        if similarity > 0.8:
            return TransferStrategy.FINE_TUNING
        elif similarity > 0.6:
            return TransferStrategy.DOMAIN_ADAPTATION
        elif similarity > 0.4:
            return TransferStrategy.FEATURE_EXTRACTION
        else:
            return TransferStrategy.PROGRESSIVE
    
    def _perform_transfer(self, source_model_id: str, target_task: Dict[str, Any], 
                         strategy: TransferStrategy) -> Dict[str, Any]:
        """Perform knowledge transfer using the selected strategy."""
        try:
            source_model = self.source_models[source_model_id]
            
            if strategy == TransferStrategy.FEATURE_EXTRACTION:
                return self._feature_extraction_transfer(source_model, target_task)
            elif strategy == TransferStrategy.FINE_TUNING:
                return self._fine_tuning_transfer(source_model, target_task)
            elif strategy == TransferStrategy.DOMAIN_ADAPTATION:
                return self._domain_adaptation_transfer(source_model, target_task)
            elif strategy == TransferStrategy.MULTI_TASK:
                return self._multi_task_transfer(source_model, target_task)
            elif strategy == TransferStrategy.PROGRESSIVE:
                return self._progressive_transfer(source_model, target_task)
            else:
                return {'error': f'Unknown transfer strategy: {strategy}'}
                
        except Exception as e:
            self.logger.error(f"Error performing transfer: {e}")
            return {'error': str(e)}
    
    def _feature_extraction_transfer(self, source_model: Dict[str, Any], 
                                   target_task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform feature extraction transfer."""
        return {
            'strategy': 'feature_extraction',
            'transferred_features': source_model['model_data'].get('features', []),
            'adaptation_required': True,
            'freeze_layers': self.config.freeze_layers
        }
    
    def _fine_tuning_transfer(self, source_model: Dict[str, Any], 
                            target_task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fine-tuning transfer."""
        return {
            'strategy': 'fine_tuning',
            'base_model': source_model['model_data'],
            'learning_rate': self.config.learning_rate_multiplier,
            'adaptation_epochs': self.config.adaptation_epochs,
            'freeze_layers': self.config.freeze_layers
        }
    
    def _domain_adaptation_transfer(self, source_model: Dict[str, Any], 
                                  target_task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform domain adaptation transfer."""
        return {
            'strategy': 'domain_adaptation',
            'source_features': source_model['model_data'].get('features', []),
            'target_features': target_task.get('features', []),
            'adaptation_layer': 'domain_adapter',
            'similarity_weight': self.config.similarity_threshold
        }
    
    def _multi_task_transfer(self, source_model: Dict[str, Any], 
                           target_task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-task transfer."""
        return {
            'strategy': 'multi_task',
            'shared_layers': source_model['model_data'].get('shared_layers', []),
            'task_specific_layers': target_task.get('task_layers', []),
            'knowledge_retention': self.config.knowledge_retention
        }
    
    def _progressive_transfer(self, source_model: Dict[str, Any], 
                            target_task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform progressive transfer."""
        return {
            'strategy': 'progressive',
            'transfer_stages': [
                'feature_extraction',
                'partial_fine_tuning',
                'full_adaptation'
            ],
            'stage_weights': [0.3, 0.4, 0.3],
            'progressive_learning_rate': self.config.learning_rate_multiplier
        }
    
    def _calculate_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """Calculate similarity between source and target domains."""
        # Simplified similarity calculation
        # In a real implementation, this would use domain-specific metrics
        if source_domain == target_domain:
            return 1.0
        
        # Check if we have previous similarity data
        if (source_domain, target_domain) in self.domain_similarities:
            return self.domain_similarities[(source_domain, target_domain)]
        
        # Calculate based on knowledge base overlap
        source_knowledge = self.knowledge_base.get(source_domain, [])
        target_knowledge = self.knowledge_base.get(target_domain, [])
        
        if not source_knowledge or not target_knowledge:
            return 0.5  # Default similarity
        
        # Calculate feature overlap
        source_features = set()
        target_features = set()
        
        for knowledge in source_knowledge:
            features = knowledge.get('experience', {}).get('features', [])
            source_features.update(features)
        
        for knowledge in target_knowledge:
            features = knowledge.get('experience', {}).get('features', [])
            target_features.update(features)
        
        if not source_features or not target_features:
            return 0.5
        
        overlap = len(source_features.intersection(target_features))
        union = len(source_features.union(target_features))
        
        return overlap / union if union > 0 else 0.5
    
    def _update_transfer_weights(self, source_domain: str, target_domain: str, 
                               performance: float) -> None:
        """Update transfer weights based on performance."""
        weight_key = f"{source_domain}->{target_domain}"
        
        if weight_key not in self.transfer_weights:
            self.transfer_weights[weight_key] = 0.5
        
        # Update weight based on performance
        self.transfer_weights[weight_key] = (
            0.9 * self.transfer_weights[weight_key] + 0.1 * performance
        )
    
    def _calculate_transfer_score(self, model: Dict[str, Any], target_domain: str) -> float:
        """Calculate transfer score for a model."""
        # Base score from model performance
        base_score = model.get('model_data', {}).get('performance', 0.5)
        
        # Weight by transfer history
        weight_key = f"{model['domain']}->{target_domain}"
        transfer_weight = self.transfer_weights.get(weight_key, 0.5)
        
        # Weight by recency
        registered_at = model.get('registered_at', datetime.now())
        recency = 1.0 / (1.0 + (datetime.now() - registered_at).days)
        
        return base_score * transfer_weight * recency
    
    def _calculate_transfer_confidence(self, source_domain: str, target_domain: str, 
                                     transfer_result: Dict[str, Any]) -> float:
        """Calculate confidence in transfer result."""
        # Base confidence from domain similarity
        similarity = self.domain_similarities.get((source_domain, target_domain), 0.5)
        
        # Adjust based on transfer strategy
        strategy = transfer_result.get('strategy', 'unknown')
        strategy_confidence = {
            'feature_extraction': 0.8,
            'fine_tuning': 0.9,
            'domain_adaptation': 0.7,
            'multi_task': 0.85,
            'progressive': 0.75
        }.get(strategy, 0.5)
        
        # Weight by historical performance
        weight_key = f"{source_domain}->{target_domain}"
        historical_weight = self.transfer_weights.get(weight_key, 0.5)
        
        return similarity * strategy_confidence * historical_weight
    
    def _estimate_performance_gain(self, transfer_result: Dict[str, Any]) -> float:
        """Estimate performance gain from transfer."""
        # Simplified estimation based on transfer strategy
        strategy = transfer_result.get('strategy', 'unknown')
        
        base_gains = {
            'feature_extraction': 0.1,
            'fine_tuning': 0.3,
            'domain_adaptation': 0.2,
            'multi_task': 0.25,
            'progressive': 0.15
        }
        
        return base_gains.get(strategy, 0.1)
    
    def _calculate_knowledge_retention(self) -> float:
        """Calculate knowledge retention rate."""
        if not self.knowledge_base:
            return 0.0
        
        total_knowledge = sum(len(knowledge) for knowledge in self.knowledge_base.values())
        if total_knowledge == 0:
            return 0.0
        
        # Calculate based on knowledge age and performance
        retention_scores = []
        for domain, knowledge in self.knowledge_base.items():
            for k in knowledge:
                age_days = (datetime.now() - k['timestamp']).days
                performance = k['performance']
                
                # Retention decreases with age but increases with performance
                retention = performance * (1.0 / (1.0 + age_days * 0.1))
                retention_scores.append(retention)
        
        return np.mean(retention_scores) if retention_scores else 0.0
