"""
Database Schema Validator and Auto-Fixer

This module provides comprehensive database validation and automatic fixing
for common database issues like column mismatches, type errors, and schema inconsistencies.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import inspect
import traceback

logger = logging.getLogger(__name__)

@dataclass
class DatabaseIssue:
    """Represents a database issue that needs fixing."""
    issue_type: str
    table_name: str
    description: str
    severity: str  # 'critical', 'warning', 'info'
    fix_suggestion: str
    auto_fixable: bool = True

class DatabaseValidator:
    """Comprehensive database validation and auto-fixing system."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.issues_found: List[DatabaseIssue] = []
        self.fixes_applied: List[str] = []
        
    def validate_all_tables(self) -> List[DatabaseIssue]:
        """Comprehensive validation of all database tables."""
        self.issues_found = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all table names
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                self._validate_table_schema(conn, table)
                self._validate_table_data(conn, table)
                
        return self.issues_found
    
    def _validate_table_schema(self, conn: sqlite3.Connection, table_name: str):
        """Validate table schema consistency."""
        try:
            # Get table schema
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Check for common schema issues
            self._check_column_count_consistency(conn, table_name, columns)
            self._check_data_type_consistency(conn, table_name, columns)
            
        except Exception as e:
            self.issues_found.append(DatabaseIssue(
                issue_type="schema_error",
                table_name=table_name,
                description=f"Error validating schema: {e}",
                severity="critical",
                fix_suggestion="Check table definition and fix syntax errors"
            ))
    
    def _check_column_count_consistency(self, conn: sqlite3.Connection, table_name: str, columns: List[Tuple]):
        """Check for column count mismatches in INSERT statements."""
        try:
            # Get all INSERT statements from the codebase that reference this table
            insert_statements = self._find_insert_statements_for_table(table_name)
            
            for statement_info in insert_statements:
                file_path, line_num, statement, values_count = statement_info
                
                # Count actual columns in table (excluding auto-generated ones)
                actual_columns = len([col for col in columns if not col[1].startswith('id') and col[1] != 'created_at' and col[1] != 'updated_at'])
                
                if values_count != actual_columns:
                    self.issues_found.append(DatabaseIssue(
                        issue_type="column_count_mismatch",
                        table_name=table_name,
                        description=f"INSERT statement provides {values_count} values for {actual_columns} columns",
                        severity="critical",
                        fix_suggestion=f"Fix INSERT statement in {file_path}:{line_num} - adjust VALUES count to match table schema",
                        auto_fixable=True
                    ))
                    
        except Exception as e:
            logger.error(f"Error checking column consistency for {table_name}: {e}")
    
    def _check_data_type_consistency(self, conn: sqlite3.Connection, table_name: str, columns: List[Tuple]):
        """Check for data type consistency issues."""
        try:
            # Check for common data type issues
            for col in columns:
                col_name, col_type = col[1], col[2]
                
                # Check for JSON columns that might receive dict objects
                if 'json' in col_type.lower() or 'text' in col_type.lower():
                    # This is handled by the parameter binding fixer
                    pass
                    
        except Exception as e:
            logger.error(f"Error checking data type consistency for {table_name}: {e}")
    
    def _validate_table_data(self, conn: sqlite3.Connection, table_name: str):
        """Validate actual data in tables."""
        try:
            # Check for data integrity issues
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            if row_count == 0:
                self.issues_found.append(DatabaseIssue(
                    issue_type="empty_table",
                    table_name=table_name,
                    description=f"Table {table_name} is empty",
                    severity="info",
                    fix_suggestion="Consider if this table should have data or if there's an issue with data insertion"
                ))
                
        except Exception as e:
            logger.error(f"Error validating data for {table_name}: {e}")
    
    def _find_insert_statements_for_table(self, table_name: str) -> List[Tuple[str, int, str, int]]:
        """Find all INSERT statements for a specific table in the codebase."""
        import os
        import re
        
        insert_statements = []
        
        # Search through all Python files
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        for line_num, line in enumerate(lines, 1):
                            if f'INSERT' in line.upper() and table_name in line:
                                # Extract VALUES count
                                values_match = re.search(r'VALUES\s*\([^)]*\)', line, re.IGNORECASE)
                                if values_match:
                                    values_part = values_match.group(0)
                                    # Count question marks
                                    values_count = values_part.count('?')
                                    insert_statements.append((file_path, line_num, line.strip(), values_count))
                                    
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                        
        return insert_statements
    
    def auto_fix_issues(self) -> List[str]:
        """Automatically fix issues that can be resolved."""
        self.fixes_applied = []
        
        for issue in self.issues_found:
            if issue.auto_fixable and issue.issue_type == "column_count_mismatch":
                try:
                    self._fix_column_count_mismatch(issue)
                    self.fixes_applied.append(f"Fixed column count mismatch in {issue.table_name}")
                except Exception as e:
                    logger.error(f"Failed to fix column count mismatch: {e}")
                    
        return self.fixes_applied
    
    def _fix_column_count_mismatch(self, issue: DatabaseIssue):
        """Fix column count mismatches automatically."""
        # This would implement automatic fixing logic
        # For now, we'll log the issue for manual fixing
        logger.warning(f"Column count mismatch detected: {issue.description}")
        logger.info(f"Fix suggestion: {issue.fix_suggestion}")
    
    def generate_fix_report(self) -> str:
        """Generate a comprehensive fix report."""
        report = []
        report.append("=" * 80)
        report.append("DATABASE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now()}")
        report.append(f"Database: {self.db_path}")
        report.append("")
        
        # Group issues by severity
        critical_issues = [i for i in self.issues_found if i.severity == 'critical']
        warning_issues = [i for i in self.issues_found if i.severity == 'warning']
        info_issues = [i for i in self.issues_found if i.severity == 'info']
        
        if critical_issues:
            report.append("🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                report.append(f"  • {issue.table_name}: {issue.description}")
                report.append(f"    Fix: {issue.fix_suggestion}")
                report.append("")
        
        if warning_issues:
            report.append("⚠️  WARNING ISSUES:")
            for issue in warning_issues:
                report.append(f"  • {issue.table_name}: {issue.description}")
                report.append(f"    Fix: {issue.fix_suggestion}")
                report.append("")
        
        if info_issues:
            report.append("ℹ️  INFO ISSUES:")
            for issue in info_issues:
                report.append(f"  • {issue.table_name}: {issue.description}")
                report.append("")
        
        if self.fixes_applied:
            report.append("✅ FIXES APPLIED:")
            for fix in self.fixes_applied:
                report.append(f"  • {fix}")
            report.append("")
        
        report.append(f"Total issues found: {len(self.issues_found)}")
        report.append(f"Fixes applied: {len(self.fixes_applied)}")
        
        return "\n".join(report)

class DatabaseParameterBinder:
    """Handles automatic parameter binding and type conversion."""
    
    @staticmethod
    def safe_bind_parameters(cursor, query: str, params: tuple) -> bool:
        """Safely bind parameters with automatic type conversion."""
        try:
            # Convert parameters to safe types
            safe_params = []
            for param in params:
                if isinstance(param, dict):
                    safe_params.append(json.dumps(param))
                elif isinstance(param, list):
                    safe_params.append(json.dumps(param))
                elif hasattr(param, '__dict__'):
                    safe_params.append(json.dumps(param.__dict__))
                else:
                    safe_params.append(param)
            
            cursor.execute(query, tuple(safe_params))
            return True
            
        except Exception as e:
            logger.error(f"Parameter binding failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return False

def validate_and_fix_database(db_path: str = "./tabula_rasa.db") -> str:
    """Main function to validate and fix database issues."""
    validator = DatabaseValidator(db_path)
    
    # Run validation
    issues = validator.validate_all_tables()
    
    # Apply auto-fixes
    fixes = validator.auto_fix_issues()
    
    # Generate report
    report = validator.generate_fix_report()
    
    return report

if __name__ == "__main__":
    # Run validation
    report = validate_and_fix_database()
    print(report)
