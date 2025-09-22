"""
Database Health Monitor

Automatically monitors database health and runs fixes on a schedule.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseHealthMonitor:
    """Monitors database health and runs automatic fixes."""
    
    def __init__(self, check_interval: int = 300):  # 5 minutes default
        self.check_interval = check_interval
        self.last_check = None
        self.health_status = "unknown"
        self.issues_found = []
        self.fixes_applied = []
        
    async def start_monitoring(self):
        """Start the database health monitoring loop."""
        logger.info("🔍 Starting database health monitoring...")
        
        while True:
            try:
                await self._run_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _run_health_check(self):
        """Run a comprehensive database health check."""
        try:
            logger.info("🔍 Running scheduled database health check...")
            
            # Run the database fix script
            result = await self._run_fix_script()
            
            if result["success"]:
                self.health_status = "healthy"
                logger.info("✅ Database health check passed")
            else:
                self.health_status = "issues_found"
                logger.warning(f"⚠️ Database health check found issues: {result['message']}")
            
            self.last_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            self.health_status = "error"
    
    async def _run_fix_script(self) -> Dict[str, Any]:
        """Run the database fix script and parse results."""
        try:
            # Run the fix script
            process = await asyncio.create_subprocess_exec(
                sys.executable, "fix_database_issues.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    "success": True,
                    "message": "Database health check completed successfully",
                    "output": stdout.decode()
                }
            else:
                return {
                    "success": False,
                    "message": f"Database health check failed: {stderr.decode()}",
                    "output": stdout.decode()
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error running fix script: {e}",
                "output": ""
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current database health status."""
        return {
            "status": self.health_status,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "check_interval": self.check_interval,
            "issues_found": len(self.issues_found),
            "fixes_applied": len(self.fixes_applied)
        }
    
    async def force_health_check(self) -> Dict[str, Any]:
        """Force an immediate health check."""
        await self._run_health_check()
        return self.get_health_status()

# Global health monitor instance
health_monitor = DatabaseHealthMonitor()

async def start_database_monitoring():
    """Start the database health monitoring system."""
    await health_monitor.start_monitoring()

async def get_database_health():
    """Get current database health status."""
    return health_monitor.get_health_status()

async def force_database_check():
    """Force an immediate database health check."""
    return await health_monitor.force_health_check()

if __name__ == "__main__":
    # Run a single health check
    async def main():
        monitor = DatabaseHealthMonitor()
        result = await monitor.force_health_check()
        print(f"Database Health Status: {result}")
    
    asyncio.run(main())
