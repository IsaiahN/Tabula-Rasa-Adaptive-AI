"""
Data Collector

Collects and aggregates data for monitoring.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import requests

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects and aggregates data for monitoring."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.scorecard_dir = os.path.join(data_dir, "scorecards")
        self.sessions_dir = os.path.join(data_dir, "sessions")
        self.logs_dir = os.path.join(data_dir, "logs")
        
        # Ensure directories exist
        os.makedirs(self.scorecard_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def collect_scorecard_data(self) -> List[Dict[str, Any]]:
        """Collect scorecard data from files."""
        try:
            scorecard_files = [f for f in os.listdir(self.scorecard_dir) if f.endswith('.json')]
            
            if not scorecard_files:
                logger.warning("No scorecard files found")
                return []
            
            # Sort by timestamp (newest first)
            scorecard_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.scorecard_dir, x)), reverse=True)
            
            scorecard_data = []
            for scorecard_file in scorecard_files:
                try:
                    file_path = os.path.join(self.scorecard_dir, scorecard_file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        scorecard_data.append(data)
                except Exception as e:
                    logger.error(f"Error reading scorecard file {scorecard_file}: {e}")
                    continue
            
            logger.info(f"Collected {len(scorecard_data)} scorecard records")
            return scorecard_data
            
        except Exception as e:
            logger.error(f"Error collecting scorecard data: {e}")
            return []
    
    def collect_session_data(self) -> List[Dict[str, Any]]:
        """Collect session data from files."""
        try:
            session_files = [f for f in os.listdir(self.sessions_dir) if f.endswith('.json')]
            
            if not session_files:
                logger.warning("No session files found")
                return []
            
            # Sort by timestamp (newest first)
            session_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.sessions_dir, x)), reverse=True)
            
            session_data = []
            for session_file in session_files:
                try:
                    file_path = os.path.join(self.sessions_dir, session_file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        session_data.append(data)
                except Exception as e:
                    logger.error(f"Error reading session file {session_file}: {e}")
                    continue
            
            logger.info(f"Collected {len(session_data)} session records")
            return session_data
            
        except Exception as e:
            logger.error(f"Error collecting session data: {e}")
            return []
    
    def collect_log_data(self) -> List[Dict[str, Any]]:
        """Collect log data from files."""
        try:
            log_files = [f for f in os.listdir(self.logs_dir) if f.endswith('.json')]
            
            if not log_files:
                logger.warning("No log files found")
                return []
            
            # Sort by timestamp (newest first)
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.logs_dir, x)), reverse=True)
            
            log_data = []
            for log_file in log_files:
                try:
                    file_path = os.path.join(self.logs_dir, log_file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        log_data.append(data)
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")
                    continue
            
            logger.info(f"Collected {len(log_data)} log records")
            return log_data
            
        except Exception as e:
            logger.error(f"Error collecting log data: {e}")
            return []
    
    def aggregate_data(self, scorecard_data: List[Dict[str, Any]], 
                      session_data: List[Dict[str, Any]], 
                      log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data from different sources."""
        try:
            aggregated = {
                'collection_timestamp': time.time(),
                'data_sources': {
                    'scorecards': len(scorecard_data),
                    'sessions': len(session_data),
                    'logs': len(log_data)
                },
                'scorecard_summary': self._summarize_scorecard_data(scorecard_data),
                'session_summary': self._summarize_session_data(session_data),
                'log_summary': self._summarize_log_data(log_data),
                'combined_metrics': self._calculate_combined_metrics(scorecard_data, session_data, log_data)
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            return {'error': str(e)}
    
    def _summarize_scorecard_data(self, scorecard_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize scorecard data."""
        try:
            if not scorecard_data:
                return {'total_records': 0}
            
            # Calculate basic metrics
            total_records = len(scorecard_data)
            successful_records = sum(1 for d in scorecard_data if d.get('success', False))
            success_rate = successful_records / total_records if total_records > 0 else 0.0
            
            # Calculate score metrics
            scores = [d.get('score', 0) for d in scorecard_data if d.get('score', 0) > 0]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            
            # Calculate time metrics
            completion_times = [d.get('completion_time', 0) for d in scorecard_data if d.get('completion_time', 0) > 0]
            avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0.0
            
            # Calculate action metrics
            actions = [d.get('actions_taken', 0) for d in scorecard_data if d.get('actions_taken', 0) > 0]
            avg_actions = sum(actions) / len(actions) if actions else 0.0
            
            return {
                'total_records': total_records,
                'success_rate': success_rate,
                'average_score': avg_score,
                'max_score': max_score,
                'average_completion_time': avg_completion_time,
                'average_actions': avg_actions
            }
            
        except Exception as e:
            logger.error(f"Error summarizing scorecard data: {e}")
            return {'error': str(e)}
    
    def _summarize_session_data(self, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize session data."""
        try:
            if not session_data:
                return {'total_records': 0}
            
            # Calculate basic metrics
            total_records = len(session_data)
            successful_sessions = sum(1 for d in session_data if d.get('success', False))
            success_rate = successful_sessions / total_records if total_records > 0 else 0.0
            
            # Calculate duration metrics
            durations = [d.get('duration', 0) for d in session_data if d.get('duration', 0) > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            
            # Calculate game distribution
            game_counts = defaultdict(int)
            for d in session_data:
                game_id = d.get('game_id', 'unknown')
                game_counts[game_id] += 1
            
            return {
                'total_records': total_records,
                'success_rate': success_rate,
                'average_duration': avg_duration,
                'game_distribution': dict(game_counts)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing session data: {e}")
            return {'error': str(e)}
    
    def _summarize_log_data(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize log data."""
        try:
            if not log_data:
                return {'total_records': 0}
            
            # Calculate basic metrics
            total_records = len(log_data)
            
            # Calculate log level distribution
            log_levels = defaultdict(int)
            for d in log_data:
                level = d.get('level', 'unknown')
                log_levels[level] += 1
            
            # Calculate error rate
            error_records = sum(1 for d in log_data if d.get('level') == 'ERROR')
            error_rate = error_records / total_records if total_records > 0 else 0.0
            
            return {
                'total_records': total_records,
                'error_rate': error_rate,
                'log_level_distribution': dict(log_levels)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing log data: {e}")
            return {'error': str(e)}
    
    def _calculate_combined_metrics(self, scorecard_data: List[Dict[str, Any]], 
                                  session_data: List[Dict[str, Any]], 
                                  log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate combined metrics from all data sources."""
        try:
            # Calculate overall success rate
            total_attempts = len(scorecard_data)
            successful_attempts = sum(1 for d in scorecard_data if d.get('success', False))
            overall_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
            
            # Calculate data freshness
            current_time = time.time()
            scorecard_freshness = self._calculate_data_freshness(scorecard_data, current_time)
            session_freshness = self._calculate_data_freshness(session_data, current_time)
            log_freshness = self._calculate_data_freshness(log_data, current_time)
            
            # Calculate data quality score
            data_quality = self._calculate_data_quality(scorecard_data, session_data, log_data)
            
            return {
                'overall_success_rate': overall_success_rate,
                'total_attempts': total_attempts,
                'data_freshness': {
                    'scorecards': scorecard_freshness,
                    'sessions': session_freshness,
                    'logs': log_freshness
                },
                'data_quality_score': data_quality
            }
            
        except Exception as e:
            logger.error(f"Error calculating combined metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_data_freshness(self, data: List[Dict[str, Any]], current_time: float) -> float:
        """Calculate data freshness score."""
        try:
            if not data:
                return 0.0
            
            # Get timestamps from data
            timestamps = []
            for d in data:
                timestamp = d.get('timestamp', 0)
                if timestamp > 0:
                    timestamps.append(timestamp)
            
            if not timestamps:
                return 0.0
            
            # Calculate average age
            avg_age = (current_time - max(timestamps)) / 3600  # Age in hours
            
            # Convert to freshness score (0-1, higher is fresher)
            if avg_age < 1:
                return 1.0
            elif avg_age < 24:
                return 0.8
            elif avg_age < 168:  # 1 week
                return 0.6
            elif avg_age < 720:  # 1 month
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Error calculating data freshness: {e}")
            return 0.0
    
    def _calculate_data_quality(self, scorecard_data: List[Dict[str, Any]], 
                               session_data: List[Dict[str, Any]], 
                               log_data: List[Dict[str, Any]]) -> float:
        """Calculate data quality score."""
        try:
            quality_scores = []
            
            # Scorecard data quality
            if scorecard_data:
                scorecard_quality = self._assess_data_quality(scorecard_data, ['game_id', 'score', 'success'])
                quality_scores.append(scorecard_quality)
            
            # Session data quality
            if session_data:
                session_quality = self._assess_data_quality(session_data, ['game_id', 'duration', 'success'])
                quality_scores.append(session_quality)
            
            # Log data quality
            if log_data:
                log_quality = self._assess_data_quality(log_data, ['level', 'message', 'timestamp'])
                quality_scores.append(log_quality)
            
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return 0.0
    
    def _assess_data_quality(self, data: List[Dict[str, Any]], required_fields: List[str]) -> float:
        """Assess quality of a dataset."""
        try:
            if not data:
                return 0.0
            
            total_records = len(data)
            valid_records = 0
            
            for record in data:
                # Check if all required fields are present and non-empty
                is_valid = True
                for field in required_fields:
                    if field not in record or record[field] is None or record[field] == '':
                        is_valid = False
                        break
                
                if is_valid:
                    valid_records += 1
            
            return valid_records / total_records if total_records > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.0
    
    def collect_recent_data(self, hours: int = 24) -> Dict[str, Any]:
        """Collect data from the last N hours."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            # Collect all data
            all_scorecard_data = self.collect_scorecard_data()
            all_session_data = self.collect_session_data()
            all_log_data = self.collect_log_data()
            
            # Filter recent data
            recent_scorecard_data = [d for d in all_scorecard_data if d.get('timestamp', 0) > cutoff_time]
            recent_session_data = [d for d in all_session_data if d.get('timestamp', 0) > cutoff_time]
            recent_log_data = [d for d in all_log_data if d.get('timestamp', 0) > cutoff_time]
            
            # Aggregate recent data
            recent_aggregated = self.aggregate_data(recent_scorecard_data, recent_session_data, recent_log_data)
            
            return {
                'time_range_hours': hours,
                'cutoff_timestamp': cutoff_time,
                'recent_data': recent_aggregated,
                'total_data': {
                    'scorecards': len(all_scorecard_data),
                    'sessions': len(all_session_data),
                    'logs': len(all_log_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting recent data: {e}")
            return {'error': str(e)}
