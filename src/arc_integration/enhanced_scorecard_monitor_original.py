#!/usr/bin/env python3
"""
Enhanced Scorecard Monitor for ARC-AGI-3 Training System

This module provides comprehensive monitoring and analysis of scorecard data
to track level completions, wins, and overall system performance.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class EnhancedScorecardMonitor:
    """
    Enhanced scorecard monitoring system that tracks:
    - Level completions and wins
    - Performance trends over time
    - Game-specific success patterns
    - Learning progress indicators
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.scorecard_dir = os.path.join(data_dir, "scorecards")
        self.sessions_dir = os.path.join(data_dir, "sessions")
        self.logs_dir = os.path.join(data_dir, "logs")
        
        # Ensure directories exist
        os.makedirs(self.scorecard_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.game_success_patterns = {}
        self.level_completion_tracking = {}
        
    async def analyze_all_scorecards(self) -> Dict[str, Any]:
        """Analyze all available scorecards for comprehensive performance tracking."""
        try:
            scorecard_files = [f for f in os.listdir(self.scorecard_dir) if f.endswith('.json')]
            
            if not scorecard_files:
                logger.warning("No scorecard files found")
                return self._create_empty_analysis()
            
            # Sort by timestamp (newest first)
            scorecard_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.scorecard_dir, x)), reverse=True)
            
            analysis = {
                'total_scorecards': len(scorecard_files),
                'analysis_timestamp': time.time(),
                'performance_summary': {},
                'level_completions': {},
                'win_rates': {},
                'trends': {},
                'recent_activity': {},
                'game_breakdown': {}
            }
            
            # Analyze recent scorecards (last 24 hours)
            recent_cutoff = time.time() - (24 * 60 * 60)
            recent_scorecards = []
            
            for scorecard_file in scorecard_files:
                file_path = os.path.join(self.scorecard_dir, scorecard_file)
                file_mtime = os.path.getmtime(file_path)
                
                if file_mtime > recent_cutoff:
                    recent_scorecards.append(scorecard_file)
            
            # Analyze recent performance
            if recent_scorecards:
                recent_analysis = self._analyze_recent_scorecards(recent_scorecards)
                analysis['recent_activity'] = recent_analysis
            
            # Analyze all scorecards for trends
            all_analysis = self._analyze_all_scorecards(scorecard_files)
            analysis.update(all_analysis)
            
            # Save analysis results
            await self._save_analysis_results(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing scorecards: {e}")
            return self._create_empty_analysis()
    
    def _analyze_recent_scorecards(self, scorecard_files: List[str]) -> Dict[str, Any]:
        """Analyze scorecards from the last 24 hours."""
        recent_analysis = {
            'scorecards_analyzed': len(scorecard_files),
            'total_wins': 0,
            'total_plays': 0,
            'level_completions': 0,
            'games_completed': 0,
            'average_score': 0.0,
            'win_rate': 0.0,
            'performance_trend': 'stable'
        }
        
        total_score = 0
        total_actions = 0
        
        for scorecard_file in scorecard_files:
            try:
                file_path = os.path.join(self.scorecard_dir, scorecard_file)
                with open(file_path, 'r') as f:
                    scorecard_data = json.load(f)
                
                # Extract performance data
                scorecard_info = scorecard_data.get('scorecard_data', {})
                analysis_info = scorecard_data.get('analysis', {})
                
                recent_analysis['total_wins'] += analysis_info.get('total_wins', 0)
                recent_analysis['total_plays'] += analysis_info.get('total_played', 0)
                recent_analysis['level_completions'] += analysis_info.get('level_completions', 0)
                recent_analysis['games_completed'] += analysis_info.get('games_completed', 0)
                
                total_score += analysis_info.get('total_score', 0)
                total_actions += analysis_info.get('total_actions', 0)
                
            except Exception as e:
                logger.warning(f"Error analyzing scorecard {scorecard_file}: {e}")
                continue
        
        # Calculate metrics
        if recent_analysis['total_plays'] > 0:
            recent_analysis['win_rate'] = (recent_analysis['total_wins'] / recent_analysis['total_plays']) * 100
            recent_analysis['average_score'] = total_score / recent_analysis['total_plays']
        
        # Determine performance trend
        if recent_analysis['win_rate'] > 50:
            recent_analysis['performance_trend'] = 'improving'
        elif recent_analysis['win_rate'] < 20:
            recent_analysis['performance_trend'] = 'declining'
        
        return recent_analysis
    
    def _analyze_all_scorecards(self, scorecard_files: List[str]) -> Dict[str, Any]:
        """Analyze all scorecards for comprehensive trends."""
        all_analysis = {
            'total_scorecards_analyzed': len(scorecard_files),
            'historical_performance': {},
            'game_success_patterns': {},
            'learning_indicators': {}
        }
        
        # Track performance over time
        performance_by_time = {}
        game_performance = {}
        
        for scorecard_file in scorecard_files:
            try:
                file_path = os.path.join(self.scorecard_dir, scorecard_file)
                with open(file_path, 'r') as f:
                    scorecard_data = json.load(f)
                
                timestamp = scorecard_data.get('timestamp', 0)
                analysis_info = scorecard_data.get('analysis', {})
                
                # Group by hour for trend analysis
                hour_key = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:00')
                if hour_key not in performance_by_time:
                    performance_by_time[hour_key] = {
                        'wins': 0, 'plays': 0, 'scores': [], 'actions': []
                    }
                
                performance_by_time[hour_key]['wins'] += analysis_info.get('total_wins', 0)
                performance_by_time[hour_key]['plays'] += analysis_info.get('total_played', 0)
                performance_by_time[hour_key]['scores'].append(analysis_info.get('total_score', 0))
                performance_by_time[hour_key]['actions'].append(analysis_info.get('total_actions', 0))
                
                # Track per-game performance
                per_game = analysis_info.get('per_game_breakdown', {})
                for game_id, game_data in per_game.items():
                    if game_id not in game_performance:
                        game_performance[game_id] = {
                            'total_wins': 0, 'total_plays': 0, 'total_score': 0,
                            'win_rate_history': [], 'score_history': []
                        }
                    
                    game_performance[game_id]['total_wins'] += game_data.get('wins', 0)
                    game_performance[game_id]['total_plays'] += game_data.get('plays', 0)
                    game_performance[game_id]['total_score'] += game_data.get('score', 0)
                    game_performance[game_id]['win_rate_history'].append(game_data.get('win_rate', 0))
                    game_performance[game_id]['score_history'].append(game_data.get('score', 0))
                
            except Exception as e:
                logger.warning(f"Error analyzing scorecard {scorecard_file}: {e}")
                continue
        
        # Calculate historical performance trends
        all_analysis['historical_performance'] = self._calculate_performance_trends(performance_by_time)
        all_analysis['game_success_patterns'] = self._analyze_game_patterns(game_performance)
        all_analysis['learning_indicators'] = self._calculate_learning_indicators(performance_by_time)
        
        return all_analysis
    
    def _calculate_performance_trends(self, performance_by_time: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        trends = {
            'win_rate_trend': 'stable',
            'score_trend': 'stable',
            'action_efficiency_trend': 'stable',
            'overall_trend': 'stable'
        }
        
        if len(performance_by_time) < 2:
            return trends
        
        # Sort by time
        sorted_times = sorted(performance_by_time.keys())
        
        # Calculate win rate trend
        win_rates = []
        for time_key in sorted_times:
            data = performance_by_time[time_key]
            if data['plays'] > 0:
                win_rate = (data['wins'] / data['plays']) * 100
                win_rates.append(win_rate)
        
        if len(win_rates) >= 2:
            if win_rates[-1] > win_rates[0] + 10:
                trends['win_rate_trend'] = 'improving'
            elif win_rates[-1] < win_rates[0] - 10:
                trends['win_rate_trend'] = 'declining'
        
        # Calculate score trend
        avg_scores = []
        for time_key in sorted_times:
            data = performance_by_time[time_key]
            if data['scores']:
                avg_scores.append(sum(data['scores']) / len(data['scores']))
        
        if len(avg_scores) >= 2:
            if avg_scores[-1] > avg_scores[0] + 5:
                trends['score_trend'] = 'improving'
            elif avg_scores[-1] < avg_scores[0] - 5:
                trends['score_trend'] = 'declining'
        
        # Determine overall trend
        improving_count = sum(1 for trend in trends.values() if trend == 'improving')
        declining_count = sum(1 for trend in trends.values() if trend == 'declining')
        
        if improving_count > declining_count:
            trends['overall_trend'] = 'improving'
        elif declining_count > improving_count:
            trends['overall_trend'] = 'declining'
        
        return trends
    
    def _analyze_game_patterns(self, game_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze success patterns for individual games."""
        patterns = {}
        
        for game_id, data in game_performance.items():
            if data['total_plays'] == 0:
                continue
            
            win_rate = (data['total_wins'] / data['total_plays']) * 100
            avg_score = data['total_score'] / data['total_plays'] if data['total_plays'] > 0 else 0
            
            patterns[game_id] = {
                'win_rate': win_rate,
                'average_score': avg_score,
                'total_attempts': data['total_plays'],
                'success_level': self._classify_success_level(win_rate, avg_score),
                'learning_progress': self._calculate_learning_progress(data['win_rate_history'])
            }
        
        return patterns
    
    def _classify_success_level(self, win_rate: float, avg_score: float) -> str:
        """Classify the success level of a game."""
        if win_rate >= 80 and avg_score >= 50:
            return 'mastery'
        elif win_rate >= 60 and avg_score >= 30:
            return 'proficient'
        elif win_rate >= 40 and avg_score >= 20:
            return 'developing'
        elif win_rate >= 20 and avg_score >= 10:
            return 'learning'
        else:
            return 'struggling'
    
    def _calculate_learning_progress(self, win_rate_history: List[float]) -> str:
        """Calculate learning progress based on win rate history."""
        if len(win_rate_history) < 3:
            return 'insufficient_data'
        
        recent_avg = sum(win_rate_history[-3:]) / 3
        early_avg = sum(win_rate_history[:3]) / 3
        
        if recent_avg > early_avg + 15:
            return 'rapid_improvement'
        elif recent_avg > early_avg + 5:
            return 'steady_improvement'
        elif recent_avg > early_avg - 5:
            return 'stable'
        else:
            return 'declining'
    
    def _calculate_learning_indicators(self, performance_by_time: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate indicators of learning progress."""
        indicators = {
            'consistency': 'unknown',
            'exploration_efficiency': 'unknown',
            'pattern_recognition': 'unknown',
            'overall_learning': 'unknown'
        }
        
        if len(performance_by_time) < 3:
            return indicators
        
        # Calculate consistency (low variance in performance)
        win_rates = []
        for data in performance_by_time.values():
            if data['plays'] > 0:
                win_rates.append((data['wins'] / data['plays']) * 100)
        
        if len(win_rates) >= 3:
            variance = sum((x - sum(win_rates)/len(win_rates))**2 for x in win_rates) / len(win_rates)
            if variance < 100:  # Low variance
                indicators['consistency'] = 'high'
            elif variance < 400:  # Medium variance
                indicators['consistency'] = 'medium'
            else:  # High variance
                indicators['consistency'] = 'low'
        
        # Calculate exploration efficiency (actions per win)
        total_actions = sum(sum(data['actions']) for data in performance_by_time.values())
        total_wins = sum(data['wins'] for data in performance_by_time.values())
        
        if total_wins > 0:
            actions_per_win = total_actions / total_wins
            if actions_per_win < 50:
                indicators['exploration_efficiency'] = 'high'
            elif actions_per_win < 100:
                indicators['exploration_efficiency'] = 'medium'
            else:
                indicators['exploration_efficiency'] = 'low'
        
        # Overall learning assessment
        improving_indicators = sum(1 for indicator in indicators.values() if indicator in ['high', 'rapid_improvement', 'steady_improvement'])
        if improving_indicators >= 3:
            indicators['overall_learning'] = 'excellent'
        elif improving_indicators >= 2:
            indicators['overall_learning'] = 'good'
        elif improving_indicators >= 1:
            indicators['overall_learning'] = 'moderate'
        else:
            indicators['overall_learning'] = 'needs_improvement'
        
        return indicators
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis when no data is available."""
        return {
            'total_scorecards': 0,
            'analysis_timestamp': time.time(),
            'performance_summary': {
                'total_wins': 0,
                'total_plays': 0,
                'win_rate': 0.0,
                'level_completions': 0,
                'games_completed': 0
            },
            'level_completions': {},
            'win_rates': {},
            'trends': {
                'overall_trend': 'no_data'
            },
            'recent_activity': {
                'scorecards_analyzed': 0,
                'performance_trend': 'no_data'
            },
            'game_breakdown': {}
        }
    
    async def _save_analysis_results(self, analysis: Dict[str, Any]):
        """Save analysis results to database."""
        try:
            integration = get_system_integration()
            
            # Log analysis to database
            await integration.log_system_event(
                level="INFO",
                component="scorecard_monitor",
                message="Scorecard analysis completed",
                data=analysis,
                session_id=f"scorecard_analysis_{int(time.time())}"
            )
            
            logger.info("Analysis results saved to database")
            
        except Exception as e:
            logger.error(f"Error saving analysis results to database: {e}")
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print a human-readable summary of the analysis."""
        print("\n" + "="*80)
        print("🎯 ENHANCED SCORECARD MONITOR - ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall performance
        perf_summary = analysis.get('performance_summary', {})
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Scorecards: {analysis.get('total_scorecards', 0)}")
        print(f"   Total Wins: {perf_summary.get('total_wins', 0)}")
        print(f"   Total Plays: {perf_summary.get('total_plays', 0)}")
        print(f"   Win Rate: {perf_summary.get('win_rate', 0.0):.1f}%")
        print(f"   Level Completions: {perf_summary.get('level_completions', 0)}")
        print(f"   Games Completed: {perf_summary.get('games_completed', 0)}")
        
        # Recent activity
        recent = analysis.get('recent_activity', {})
        if recent.get('scorecards_analyzed', 0) > 0:
            print(f"\n🔥 RECENT ACTIVITY (Last 24h):")
            print(f"   Scorecards Analyzed: {recent.get('scorecards_analyzed', 0)}")
            print(f"   Recent Wins: {recent.get('total_wins', 0)}")
            print(f"   Recent Plays: {recent.get('total_plays', 0)}")
            print(f"   Recent Win Rate: {recent.get('win_rate', 0.0):.1f}%")
            print(f"   Performance Trend: {recent.get('performance_trend', 'unknown').upper()}")
        
        # Trends
        trends = analysis.get('trends', {})
        print(f"\n📈 PERFORMANCE TRENDS:")
        print(f"   Win Rate Trend: {trends.get('win_rate_trend', 'unknown').upper()}")
        print(f"   Score Trend: {trends.get('score_trend', 'unknown').upper()}")
        print(f"   Overall Trend: {trends.get('overall_trend', 'unknown').upper()}")
        
        # Learning indicators
        learning = analysis.get('learning_indicators', {})
        if learning:
            print(f"\n🧠 LEARNING INDICATORS:")
            print(f"   Consistency: {learning.get('consistency', 'unknown').upper()}")
            print(f"   Exploration Efficiency: {learning.get('exploration_efficiency', 'unknown').upper()}")
            print(f"   Overall Learning: {learning.get('overall_learning', 'unknown').upper()}")
        
        # Game breakdown
        game_patterns = analysis.get('game_success_patterns', {})
        if game_patterns:
            print(f"\n🎮 GAME-SPECIFIC PERFORMANCE:")
            for game_id, pattern in list(game_patterns.items())[:5]:  # Show top 5 games
                print(f"   {game_id}: {pattern.get('success_level', 'unknown').upper()} "
                      f"(Win Rate: {pattern.get('win_rate', 0.0):.1f}%, "
                      f"Learning: {pattern.get('learning_progress', 'unknown').upper()})")
        
        print("\n" + "="*80)

def main():
    """Main function for running the enhanced scorecard monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Scorecard Monitor for ARC-AGI-3')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    monitor = EnhancedScorecardMonitor(args.data_dir)
    analysis = monitor.analyze_all_scorecards()
    monitor.print_analysis_summary(analysis)
    
    return analysis

if __name__ == "__main__":
    main()
