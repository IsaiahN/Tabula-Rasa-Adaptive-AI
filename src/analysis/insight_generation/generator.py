"""
Insight Generator

Generates insights and recommendations from analysis results.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class InsightGenerator:
    """Generates insights and recommendations from analysis results."""
    
    def __init__(self):
        self.insight_templates = self._initialize_insight_templates()
    
    def _initialize_insight_templates(self) -> Dict[str, str]:
        """Initialize templates for generating insights."""
        return {
            'high_success_sequence': "High success sequence found: {sequence} with {success_rate:.1%} success rate",
            'effective_action': "Action '{action}' is highly effective with {success_rate:.1%} success rate",
            'coordinate_pattern': "Coordinate pattern detected: {pattern} with {effectiveness:.1%} effectiveness",
            'game_specific': "Game '{game_id}' shows {insight}",
            'trend_improvement': "Performance is improving: {metric} increased by {change:.1%}",
            'trend_decline': "Performance is declining: {metric} decreased by {change:.1%}",
            'recommendation': "Recommendation: {recommendation}"
        }
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from analysis results."""
        try:
            insights = []
            recommendations = []
            
            # Generate insights from different analysis components
            insights.extend(self._generate_sequence_insights(analysis_results))
            insights.extend(self._generate_action_insights(analysis_results))
            insights.extend(self._generate_coordinate_insights(analysis_results))
            insights.extend(self._generate_game_insights(analysis_results))
            insights.extend(self._generate_trend_insights(analysis_results))
            
            # Generate recommendations
            recommendations.extend(self._generate_recommendations(analysis_results))
            
            # Calculate insight confidence scores
            insights_with_confidence = self._calculate_insight_confidence(insights)
            
            return {
                'insights': insights_with_confidence,
                'recommendations': recommendations,
                'total_insights': len(insights),
                'high_confidence_insights': len([i for i in insights_with_confidence if i.get('confidence', 0) > 0.8])
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {'insights': [], 'recommendations': [], 'error': str(e)}
    
    def _generate_sequence_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from sequence analysis."""
        insights = []
        
        try:
            # Get sequence data
            successful_sequences = analysis_results.get('successful_sequences', {})
            frequent_sequences = analysis_results.get('frequent_sequences', {})
            
            # High success sequences
            if 'high_success_sequences' in successful_sequences:
                for seq_data in successful_sequences['high_success_sequences'][:5]:
                    insight = {
                        'type': 'high_success_sequence',
                        'content': self.insight_templates['high_success_sequence'].format(
                            sequence=seq_data['sequence'],
                            success_rate=seq_data['success_rate']
                        ),
                        'data': seq_data,
                        'priority': 'high'
                    }
                    insights.append(insight)
            
            # Common sequences
            if 'common_sequences' in successful_sequences:
                for seq_data in successful_sequences['common_sequences'][:3]:
                    insight = {
                        'type': 'common_sequence',
                        'content': f"Common sequence: {seq_data['sequence']} appears {seq_data['count']} times",
                        'data': seq_data,
                        'priority': 'medium'
                    }
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating sequence insights: {e}")
        
        return insights
    
    def _generate_action_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from action analysis."""
        insights = []
        
        try:
            # Get action effectiveness data
            action_effectiveness = analysis_results.get('action_effectiveness', {})
            
            if 'action_effectiveness' in action_effectiveness:
                actions = action_effectiveness['action_effectiveness']
                
                # Most effective action
                if actions:
                    most_effective = max(actions.items(), key=lambda x: x[1]['effectiveness_score'])
                    insight = {
                        'type': 'effective_action',
                        'content': self.insight_templates['effective_action'].format(
                            action=most_effective[0],
                            success_rate=most_effective[1]['success_rate']
                        ),
                        'data': most_effective[1],
                        'priority': 'high'
                    }
                    insights.append(insight)
                
                # Least effective action
                least_effective = min(actions.items(), key=lambda x: x[1]['effectiveness_score'])
                if least_effective[1]['effectiveness_score'] < 0.3:
                    insight = {
                        'type': 'ineffective_action',
                        'content': f"Action '{least_effective[0]}' has low effectiveness: {least_effective[1]['effectiveness_score']:.1%}",
                        'data': least_effective[1],
                        'priority': 'medium'
                    }
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating action insights: {e}")
        
        return insights
    
    def _generate_coordinate_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from coordinate analysis."""
        insights = []
        
        try:
            # Get coordinate patterns
            coordinate_patterns = analysis_results.get('coordinate_patterns', {})
            
            # Coordinate clusters
            if 'coordinate_clusters' in coordinate_patterns:
                clusters = coordinate_patterns['coordinate_clusters']
                for cluster in clusters[:3]:  # Top 3 clusters
                    insight = {
                        'type': 'coordinate_cluster',
                        'content': f"Coordinate cluster at {cluster['center']} with {cluster['point_count']} points and {cluster['success_rate']:.1%} success rate",
                        'data': cluster,
                        'priority': 'medium'
                    }
                    insights.append(insight)
            
            # Coordinate effectiveness
            if 'coordinate_effectiveness' in coordinate_patterns:
                coord_eff = coordinate_patterns['coordinate_effectiveness']
                for action, data in coord_eff.items():
                    if data['success_rate'] > 0.8:
                        insight = {
                            'type': 'coordinate_pattern',
                            'content': self.insight_templates['coordinate_pattern'].format(
                                pattern=f"{action} at {data['coordinate_center']}",
                                effectiveness=data['success_rate']
                            ),
                            'data': data,
                            'priority': 'high'
                        }
                        insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating coordinate insights: {e}")
        
        return insights
    
    def _generate_game_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from game-specific analysis."""
        insights = []
        
        try:
            # Get game-specific patterns
            game_patterns = analysis_results.get('game_specific_patterns', {})
            
            for game_id, patterns in game_patterns.items():
                if patterns.get('total_traces', 0) > 5:  # Only analyze games with sufficient data
                    success_rate = patterns.get('success_rate', 0.0)
                    
                    if success_rate > 0.8:
                        insight = {
                            'type': 'game_specific',
                            'content': f"Game '{game_id}' has high success rate: {success_rate:.1%}",
                            'data': patterns,
                            'priority': 'high'
                        }
                        insights.append(insight)
                    elif success_rate < 0.3:
                        insight = {
                            'type': 'game_specific',
                            'content': f"Game '{game_id}' has low success rate: {success_rate:.1%}",
                            'data': patterns,
                            'priority': 'medium'
                        }
                        insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating game insights: {e}")
        
        return insights
    
    def _generate_trend_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from trend analysis."""
        insights = []
        
        try:
            # Get trends data
            trends = analysis_results.get('trends', {})
            
            if 'success_rate_trend' in trends:
                trend = trends['success_rate_trend']
                if trend == 'improving':
                    change = trends.get('success_trend_strength', 0.0)
                    insight = {
                        'type': 'trend_improvement',
                        'content': self.insight_templates['trend_improvement'].format(
                            metric='success rate',
                            change=change * 100
                        ),
                        'data': trends,
                        'priority': 'high'
                    }
                    insights.append(insight)
                elif trend == 'declining':
                    change = trends.get('success_trend_strength', 0.0)
                    insight = {
                        'type': 'trend_decline',
                        'content': self.insight_templates['trend_decline'].format(
                            metric='success rate',
                            change=change * 100
                        ),
                        'data': trends,
                        'priority': 'high'
                    }
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating trend insights: {e}")
        
        return insights
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        try:
            # Get overall metrics
            overall_metrics = analysis_results.get('overall_metrics', {})
            success_rate = overall_metrics.get('success_rate', 0.0)
            
            # Success rate recommendations
            if success_rate < 0.3:
                recommendations.append({
                    'type': 'success_rate',
                    'content': "Success rate is very low. Consider focusing on high-success sequences and effective actions.",
                    'priority': 'high'
                })
            elif success_rate < 0.6:
                recommendations.append({
                    'type': 'success_rate',
                    'content': "Success rate is moderate. Try to identify and replicate successful patterns.",
                    'priority': 'medium'
                })
            
            # Action effectiveness recommendations
            action_effectiveness = analysis_results.get('action_effectiveness', {})
            if 'action_effectiveness' in action_effectiveness:
                actions = action_effectiveness['action_effectiveness']
                if actions:
                    # Find underutilized effective actions
                    for action, data in actions.items():
                        if data['effectiveness_score'] > 0.7 and data['total_usage'] < 10:
                            recommendations.append({
                                'type': 'action_usage',
                                'content': f"Action '{action}' is highly effective but underutilized. Consider using it more frequently.",
                                'priority': 'medium'
                            })
            
            # Sequence recommendations
            successful_sequences = analysis_results.get('successful_sequences', {})
            if 'high_success_sequences' in successful_sequences:
                high_success_seqs = successful_sequences['high_success_sequences']
                if high_success_seqs:
                    best_seq = high_success_seqs[0]
                    recommendations.append({
                        'type': 'sequence_usage',
                        'content': f"Use the high-success sequence {best_seq['sequence']} more frequently (success rate: {best_seq['success_rate']:.1%})",
                        'priority': 'high'
                    })
            
            # Game-specific recommendations
            game_metrics = analysis_results.get('game_metrics', {})
            low_performance_games = [game_id for game_id, data in game_metrics.items() 
                                   if data.get('success_rate', 0) < 0.4]
            
            if low_performance_games:
                recommendations.append({
                    'type': 'game_focus',
                    'content': f"Focus on improving performance in games: {', '.join(low_performance_games[:3])}",
                    'priority': 'medium'
                })
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_insight_confidence(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate confidence scores for insights."""
        try:
            insights_with_confidence = []
            
            for insight in insights:
                confidence = 0.5  # Base confidence
                
                # Adjust confidence based on data quality
                data = insight.get('data', {})
                
                # Higher confidence for insights with more data points
                if 'count' in data:
                    confidence += min(0.3, data['count'] / 100)
                
                if 'success_rate' in data:
                    # Higher confidence for extreme success rates
                    if data['success_rate'] > 0.9 or data['success_rate'] < 0.1:
                        confidence += 0.2
                
                if 'effectiveness_score' in data:
                    # Higher confidence for high effectiveness scores
                    if data['effectiveness_score'] > 0.8:
                        confidence += 0.2
                
                # Priority affects confidence
                priority = insight.get('priority', 'medium')
                if priority == 'high':
                    confidence += 0.1
                elif priority == 'low':
                    confidence -= 0.1
                
                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                
                insight['confidence'] = confidence
                insights_with_confidence.append(insight)
            
            return insights_with_confidence
            
        except Exception as e:
            logger.error(f"Error calculating insight confidence: {e}")
            return insights
