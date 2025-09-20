"""
Report Generator

Generates monitoring reports and summaries.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates monitoring reports and summaries."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_performance_report(self, performance_data: Dict[str, Any], 
                                  trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        try:
            report = {
                'report_type': 'performance',
                'generated_at': datetime.now().isoformat(),
                'summary': self._generate_summary(performance_data, trend_data),
                'detailed_metrics': performance_data,
                'trend_analysis': trend_data,
                'recommendations': self._generate_recommendations(performance_data, trend_data),
                'charts': self._generate_chart_data(performance_data, trend_data)
            }
            
            # Save report to file
            report_filename = f"performance_report_{int(time.time())}.json"
            report_path = os.path.join(self.output_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def generate_trend_report(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trend analysis report."""
        try:
            report = {
                'report_type': 'trend_analysis',
                'generated_at': datetime.now().isoformat(),
                'trend_summary': trend_data.get('trend_summary', {}),
                'success_trends': trend_data.get('success_trends', {}),
                'score_trends': trend_data.get('score_trends', {}),
                'time_trends': trend_data.get('time_trends', {}),
                'action_trends': trend_data.get('action_trends', {}),
                'insights': self._generate_trend_insights(trend_data)
            }
            
            # Save report to file
            report_filename = f"trend_report_{int(time.time())}.json"
            report_path = os.path.join(self.output_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Trend report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating trend report: {e}")
            return {'error': str(e)}
    
    def generate_game_report(self, game_id: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a game-specific report."""
        try:
            report = {
                'report_type': 'game_analysis',
                'game_id': game_id,
                'generated_at': datetime.now().isoformat(),
                'game_summary': self._generate_game_summary(game_data),
                'performance_metrics': game_data,
                'recommendations': self._generate_game_recommendations(game_data)
            }
            
            # Save report to file
            report_filename = f"game_report_{game_id}_{int(time.time())}.json"
            report_path = os.path.join(self.output_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Game report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating game report: {e}")
            return {'error': str(e)}
    
    def _generate_summary(self, performance_data: Dict[str, Any], 
                         trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of performance data."""
        try:
            overall_metrics = performance_data.get('overall_metrics', {})
            trend_summary = trend_data.get('trend_summary', {})
            
            summary = {
                'total_attempts': overall_metrics.get('total_attempts', 0),
                'success_rate': overall_metrics.get('success_rate', 0.0),
                'average_score': overall_metrics.get('average_score', 0.0),
                'overall_trend': trend_summary.get('overall_trend', 'unknown'),
                'trend_confidence': trend_summary.get('trend_confidence', 0.0),
                'performance_grade': self._calculate_performance_grade(overall_metrics, trend_summary)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
    
    def _generate_recommendations(self, performance_data: Dict[str, Any], 
                                 trend_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on performance and trend data."""
        try:
            recommendations = []
            
            overall_metrics = performance_data.get('overall_metrics', {})
            trend_summary = trend_data.get('trend_summary', {})
            
            # Success rate recommendations
            success_rate = overall_metrics.get('success_rate', 0.0)
            if success_rate < 0.3:
                recommendations.append({
                    'type': 'success_rate',
                    'priority': 'high',
                    'title': 'Low Success Rate',
                    'description': f'Success rate is very low ({success_rate:.1%}). Focus on fundamental strategies and high-confidence patterns.',
                    'action': 'Review and implement proven strategies from successful games'
                })
            elif success_rate < 0.6:
                recommendations.append({
                    'type': 'success_rate',
                    'priority': 'medium',
                    'title': 'Moderate Success Rate',
                    'description': f'Success rate is moderate ({success_rate:.1%}). Identify and replicate successful patterns.',
                    'action': 'Analyze high-performing games and apply similar strategies'
                })
            
            # Trend-based recommendations
            overall_trend = trend_summary.get('overall_trend', 'unknown')
            if overall_trend == 'declining':
                recommendations.append({
                    'type': 'trend',
                    'priority': 'high',
                    'title': 'Declining Performance',
                    'description': 'Performance is declining. Immediate action needed.',
                    'action': 'Review recent changes and revert to previously successful strategies'
                })
            elif overall_trend == 'improving':
                recommendations.append({
                    'type': 'trend',
                    'priority': 'low',
                    'title': 'Improving Performance',
                    'description': 'Performance is improving. Continue current strategies.',
                    'action': 'Maintain current approach and monitor for continued improvement'
                })
            
            # Score-based recommendations
            avg_score = overall_metrics.get('average_score', 0.0)
            if avg_score < 50:
                recommendations.append({
                    'type': 'score',
                    'priority': 'medium',
                    'title': 'Low Average Score',
                    'description': f'Average score is low ({avg_score:.1f}). Focus on score optimization.',
                    'action': 'Analyze high-scoring games and implement score-maximizing strategies'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _generate_trend_insights(self, trend_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from trend data."""
        try:
            insights = []
            
            success_trends = trend_data.get('success_trends', {})
            score_trends = trend_data.get('score_trends', {})
            time_trends = trend_data.get('time_trends', {})
            action_trends = trend_data.get('action_trends', {})
            
            # Success rate insights
            if success_trends.get('direction') == 'improving':
                insights.append({
                    'type': 'positive',
                    'title': 'Success Rate Improving',
                    'description': f"Success rate is improving with {success_trends.get('strength', 0):.1%} strength",
                    'confidence': success_trends.get('strength', 0.0)
                })
            elif success_trends.get('direction') == 'declining':
                insights.append({
                    'type': 'negative',
                    'title': 'Success Rate Declining',
                    'description': f"Success rate is declining with {success_trends.get('strength', 0):.1%} strength",
                    'confidence': success_trends.get('strength', 0.0)
                })
            
            # Score insights
            if score_trends.get('direction') == 'improving':
                insights.append({
                    'type': 'positive',
                    'title': 'Scores Improving',
                    'description': f"Average scores are improving with {score_trends.get('strength', 0):.1%} strength",
                    'confidence': score_trends.get('strength', 0.0)
                })
            
            # Time insights
            if time_trends.get('direction') == 'improving':
                insights.append({
                    'type': 'positive',
                    'title': 'Completion Time Improving',
                    'description': f"Completion times are improving (getting faster) with {time_trends.get('strength', 0):.1%} strength",
                    'confidence': time_trends.get('strength', 0.0)
                })
            
            # Action insights
            if action_trends.get('direction') == 'improving':
                insights.append({
                    'type': 'positive',
                    'title': 'Action Efficiency Improving',
                    'description': f"Action efficiency is improving (fewer actions needed) with {action_trends.get('strength', 0):.1%} strength",
                    'confidence': action_trends.get('strength', 0.0)
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating trend insights: {e}")
            return []
    
    def _generate_game_summary(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary for a specific game."""
        try:
            return {
                'total_attempts': game_data.get('total_attempts', 0),
                'success_rate': game_data.get('success_rate', 0.0),
                'recent_success_rate': game_data.get('recent_success_rate', 0.0),
                'levels_completed': game_data.get('levels_completed', 0),
                'max_level': game_data.get('max_level', 0),
                'average_score': game_data.get('average_score', 0.0),
                'performance_trend': game_data.get('performance_trend', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error generating game summary: {e}")
            return {}
    
    def _generate_game_recommendations(self, game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for a specific game."""
        try:
            recommendations = []
            
            success_rate = game_data.get('success_rate', 0.0)
            recent_success_rate = game_data.get('recent_success_rate', 0.0)
            performance_trend = game_data.get('performance_trend', 'unknown')
            
            # Success rate recommendations
            if success_rate < 0.3:
                recommendations.append({
                    'type': 'success_rate',
                    'priority': 'high',
                    'title': 'Low Success Rate',
                    'description': f'Success rate is very low ({success_rate:.1%}). Focus on basic strategies.',
                    'action': 'Start with simple, proven patterns for this game'
                })
            
            # Recent performance recommendations
            if recent_success_rate > success_rate:
                recommendations.append({
                    'type': 'recent_performance',
                    'priority': 'medium',
                    'title': 'Recent Improvement',
                    'description': f'Recent performance ({recent_success_rate:.1%}) is better than overall ({success_rate:.1%}).',
                    'action': 'Continue with recent strategies that are working'
                })
            
            # Trend-based recommendations
            if performance_trend == 'improving':
                recommendations.append({
                    'type': 'trend',
                    'priority': 'low',
                    'title': 'Improving Trend',
                    'description': 'Performance is improving for this game.',
                    'action': 'Maintain current approach'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating game recommendations: {e}")
            return []
    
    def _generate_chart_data(self, performance_data: Dict[str, Any], 
                           trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for charts and visualizations."""
        try:
            chart_data = {
                'success_rate_over_time': trend_data.get('success_trends', {}).get('rolling_rates', []),
                'score_over_time': trend_data.get('score_trends', {}).get('rolling_scores', []),
                'completion_time_over_time': trend_data.get('time_trends', {}).get('rolling_times', []),
                'actions_over_time': trend_data.get('action_trends', {}).get('rolling_actions', [])
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            return {}
    
    def _calculate_performance_grade(self, overall_metrics: Dict[str, Any], 
                                   trend_summary: Dict[str, Any]) -> str:
        """Calculate a performance grade based on metrics and trends."""
        try:
            success_rate = overall_metrics.get('success_rate', 0.0)
            avg_score = overall_metrics.get('average_score', 0.0)
            overall_trend = trend_summary.get('overall_trend', 'unknown')
            
            # Base grade on success rate
            if success_rate >= 0.8:
                base_grade = 'A'
            elif success_rate >= 0.6:
                base_grade = 'B'
            elif success_rate >= 0.4:
                base_grade = 'C'
            elif success_rate >= 0.2:
                base_grade = 'D'
            else:
                base_grade = 'F'
            
            # Adjust based on trend
            if overall_trend == 'improving':
                if base_grade == 'A':
                    return 'A+'
                elif base_grade == 'B':
                    return 'A'
                elif base_grade == 'C':
                    return 'B'
                elif base_grade == 'D':
                    return 'C'
                else:
                    return 'D'
            elif overall_trend == 'declining':
                if base_grade == 'A':
                    return 'B'
                elif base_grade == 'B':
                    return 'C'
                elif base_grade == 'C':
                    return 'D'
                elif base_grade == 'D':
                    return 'F'
                else:
                    return 'F'
            else:
                return base_grade
                
        except Exception as e:
            logger.error(f"Error calculating performance grade: {e}")
            return 'Unknown'
