"""
Learned Patterns Manager
Replaces learned_patterns.pkl with database storage.
"""

import hashlib
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import sqlite3
from pathlib import Path

class LearnedPatternsManager:
    """
    Manages learned patterns in the database.
    Replaces the learned_patterns.pkl file.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure the learned_patterns table exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT NOT NULL UNIQUE,
                    pattern_data TEXT NOT NULL,
                    pattern_type TEXT,
                    success_rate REAL,
                    usage_count INTEGER DEFAULT 1,
                    last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _generate_pattern_hash(self, pattern_data: Dict[str, Any]) -> str:
        """Generate a hash for pattern deduplication."""
        # Create a stable representation for hashing
        stable_data = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(stable_data.encode()).hexdigest()
    
    def add_pattern(self, 
                   pattern_data: Dict[str, Any], 
                   pattern_type: str = None,
                   success_rate: float = 0.5) -> Dict[str, Any]:
        """
        Add a learned pattern.
        
        Args:
            pattern_data: The pattern data
            pattern_type: Type of pattern
            success_rate: Success rate of the pattern
            
        Returns:
            Dict with pattern info and whether it was new
        """
        pattern_hash = self._generate_pattern_hash(pattern_data)
        pattern_json = json.dumps(pattern_data)
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if pattern already exists
            cursor = conn.execute("""
                SELECT id, usage_count, success_rate, last_used
                FROM learned_patterns 
                WHERE pattern_hash = ?
            """, (pattern_hash,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                pattern_id, usage_count, old_success_rate, last_used = existing
                
                # Update success rate with weighted average
                new_success_rate = (old_success_rate + success_rate) / 2
                
                conn.execute("""
                    UPDATE learned_patterns 
                    SET usage_count = usage_count + 1,
                        success_rate = ?,
                        last_used = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_success_rate, pattern_id))
                
                return {
                    'is_new': False,
                    'pattern_id': pattern_id,
                    'usage_count': usage_count + 1,
                    'success_rate': new_success_rate,
                    'last_used': datetime.now().isoformat(),
                    'message': f"Pattern updated (usage #{usage_count + 1})"
                }
            else:
                # Insert new pattern
                cursor = conn.execute("""
                    INSERT INTO learned_patterns 
                    (pattern_hash, pattern_data, pattern_type, success_rate, usage_count)
                    VALUES (?, ?, ?, ?, 1)
                """, (pattern_hash, pattern_json, pattern_type, success_rate))
                
                pattern_id = cursor.lastrowid
                conn.commit()
                
                return {
                    'is_new': True,
                    'pattern_id': pattern_id,
                    'usage_count': 1,
                    'success_rate': success_rate,
                    'last_used': datetime.now().isoformat(),
                    'message': "New pattern added"
                }
    
    def get_patterns(self, 
                    pattern_type: str = None,
                    min_success_rate: float = 0.0,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get learned patterns with optional filtering.
        
        Args:
            pattern_type: Filter by pattern type
            min_success_rate: Minimum success rate
            limit: Maximum number of patterns to return
            
        Returns:
            List of pattern dictionaries
        """
        query = """
            SELECT id, pattern_data, pattern_type, success_rate, 
                   usage_count, last_used, created_at
            FROM learned_patterns
            WHERE success_rate >= ?
        """
        params = [min_success_rate]
        
        if pattern_type:
            query += " AND pattern_type = ?"
            params.append(pattern_type)
        
        query += " ORDER BY success_rate DESC, usage_count DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'id': row[0],
                    'pattern_data': json.loads(row[1]),
                    'pattern_type': row[2],
                    'success_rate': row[3],
                    'usage_count': row[4],
                    'last_used': row[5],
                    'created_at': row[6]
                })
            
            return patterns
    
    def find_similar_patterns(self, 
                             pattern_data: Dict[str, Any], 
                             threshold: float = 0.8,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find patterns similar to the given pattern.
        
        Args:
            pattern_data: Pattern to find similarities for
            threshold: Similarity threshold (0-1)
            limit: Maximum number of similar patterns
            
        Returns:
            List of similar patterns
        """
        # This is a simplified similarity search
        # In a real implementation, you might use more sophisticated similarity algorithms
        
        patterns = self.get_patterns(limit=limit * 2)  # Get more to filter
        similar_patterns = []
        
        for pattern in patterns:
            # Simple similarity based on shared keys and values
            similarity = self._calculate_similarity(pattern_data, pattern['pattern_data'])
            if similarity >= threshold:
                pattern['similarity'] = similarity
                similar_patterns.append(pattern)
        
        # Sort by similarity and return top results
        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_patterns[:limit]
    
    def _calculate_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns."""
        if not pattern1 or not pattern2:
            return 0.0
        
        # Get all keys from both patterns
        keys1 = set(pattern1.keys())
        keys2 = set(pattern2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        # Calculate Jaccard similarity for keys
        intersection = len(keys1.intersection(keys2))
        union = len(keys1.union(keys2))
        key_similarity = intersection / union if union > 0 else 0.0
        
        # Calculate value similarity for common keys
        common_keys = keys1.intersection(keys2)
        value_similarity = 0.0
        
        if common_keys:
            for key in common_keys:
                if pattern1[key] == pattern2[key]:
                    value_similarity += 1.0
            value_similarity /= len(common_keys)
        
        # Combine key and value similarity
        return (key_similarity + value_similarity) / 2
    
    def update_pattern_success(self, pattern_id: int, success: bool) -> bool:
        """
        Update pattern success rate based on outcome.
        
        Args:
            pattern_id: ID of the pattern
            success: Whether the pattern was successful
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current success rate
                cursor = conn.execute("""
                    SELECT success_rate, usage_count FROM learned_patterns 
                    WHERE id = ?
                """, (pattern_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                current_rate, usage_count = result
                
                # Update with weighted average
                success_value = 1.0 if success else 0.0
                new_rate = (current_rate * usage_count + success_value) / (usage_count + 1)
                
                conn.execute("""
                    UPDATE learned_patterns 
                    SET success_rate = ?,
                        usage_count = usage_count + 1,
                        last_used = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_rate, pattern_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error updating pattern success: {e}")
            return False
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_patterns,
                    COUNT(DISTINCT pattern_type) as pattern_types,
                    AVG(success_rate) as avg_success_rate,
                    MAX(success_rate) as max_success_rate,
                    MIN(success_rate) as min_success_rate,
                    SUM(usage_count) as total_usage
                FROM learned_patterns
            """)
            
            stats = cursor.fetchone()
            
            # Get pattern type distribution
            cursor = conn.execute("""
                SELECT pattern_type, COUNT(*) as count, AVG(success_rate) as avg_rate
                FROM learned_patterns
                GROUP BY pattern_type
                ORDER BY count DESC
            """)
            
            type_distribution = [
                {'type': row[0] or 'unknown', 'count': row[1], 'avg_success_rate': row[2]}
                for row in cursor.fetchall()
            ]
            
            return {
                'total_patterns': stats[0] or 0,
                'pattern_types': stats[1] or 0,
                'avg_success_rate': stats[2] or 0,
                'max_success_rate': stats[3] or 0,
                'min_success_rate': stats[4] or 0,
                'total_usage': stats[5] or 0,
                'type_distribution': type_distribution
            }
    
    def migrate_from_pickle(self, file_path: str = "data/learned_patterns.pkl") -> bool:
        """
        Migrate patterns from the old pickle file to database.
        
        Args:
            file_path: Path to the old pickle file
            
        Returns:
            True if migration successful
        """
        try:
            if not Path(file_path).exists():
                print(f"File {file_path} does not exist, skipping migration")
                return True
            
            with open(file_path, 'rb') as f:
                patterns = pickle.load(f)
            
            if not isinstance(patterns, dict):
                print(f"Invalid pickle file format: {file_path}")
                return False
            
            # Migrate patterns
            migrated_count = 0
            for pattern_id, pattern_data in patterns.items():
                if isinstance(pattern_data, dict):
                    self.add_pattern(pattern_data, success_rate=0.5)
                    migrated_count += 1
            
            print(f"Successfully migrated {migrated_count} patterns from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error migrating patterns from pickle: {e}")
            return False
    
    def cleanup_old_patterns(self, days_old: int = 30, min_usage: int = 1) -> int:
        """Remove old, unused patterns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM learned_patterns 
                WHERE last_used < datetime('now', '-{} days')
                AND usage_count < ?
            """.format(days_old), (min_usage,))
            
            return cursor.rowcount

# Global instance
_learned_patterns_manager = None

def get_learned_patterns_manager() -> LearnedPatternsManager:
    """Get the global learned patterns manager instance."""
    global _learned_patterns_manager
    if _learned_patterns_manager is None:
        _learned_patterns_manager = LearnedPatternsManager()
    return _learned_patterns_manager
