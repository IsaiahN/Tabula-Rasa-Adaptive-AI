#!/usr/bin/env python3
"""
DIRECTOR PERMANENT INITIALIZATION
Comprehensive startup script that combines all Director initialization tasks
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database.director_commands import get_director_commands
from database.system_integration import get_system_integration

class DirectorInitializer:
    """Permanent Director initialization and analysis system."""
    
    def __init__(self):
        self.director = get_director_commands()
        self.integration = get_system_integration()
        self.session_id = f"director_init_{int(datetime.now().timestamp())}"
    
    async def initialize_director(self):
        """Complete Director initialization process."""
        print("🎯 DIRECTOR PERMANENT INITIALIZATION")
        print("=" * 60)
        
        try:
            # 1. Retrieve and analyze self-model
            print("\n📚 Analyzing self-model entries...")
            self_model_entries = await self.integration.get_self_model_entries(limit=50)
            print(f"   Found {len(self_model_entries)} self-model entries")
            
            # 2. Get comprehensive system overview
            print("\n🔍 Analyzing system overview...")
            system_overview = await self.director.get_system_overview()
            print(f"   System Status: {system_overview.get('system_status', {})}")
            
            # 3. Analyze learning patterns
            print("\n🧠 Analyzing learning patterns...")
            learning_analysis = await self.director.get_learning_analysis()
            print(f"   Learning Analysis: {learning_analysis.get('summary', {})}")
            
            # 4. Check system health
            print("\n🏥 Checking system health...")
            system_health = await self.director.analyze_system_health()
            print(f"   System Health: {system_health.get('overall_status', 'Unknown')}")
            
            # 5. Analyze action effectiveness
            print("\n⚡ Analyzing action effectiveness...")
            action_effectiveness = await self.director.get_action_effectiveness()
            print(f"   Action Effectiveness: {len(action_effectiveness.get('patterns', []))} patterns found")
            
            # 6. Analyze coordinate intelligence
            print("\n🎯 Analyzing coordinate intelligence...")
            coordinate_intelligence = await self.director.get_coordinate_intelligence()
            print(f"   Coordinate Intelligence: {len(coordinate_intelligence.get('patterns', []))} patterns found")
            
            # 7. Test database connectivity
            print("\n🔍 Testing database connectivity...")
            connectivity_results = await self._test_database_connectivity()
            print(f"   Database Tests: {connectivity_results['passed']}/{connectivity_results['total']} passed")
            
            # 8. Add current session reflection
            print("\n🧠 Adding session reflection...")
            await self._add_session_reflection(system_overview, learning_analysis, system_health)
            
            # 9. Generate summary
            print("\n📊 Generating system summary...")
            summary = self._generate_summary(system_overview, learning_analysis, system_health, connectivity_results)
            
            print("\n✅ DIRECTOR INITIALIZATION COMPLETE!")
            print("=" * 60)
            print(summary)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            return False
    
    async def _test_database_connectivity(self):
        """Test all database API functions."""
        tests_passed = 0
        total_tests = 0
        
        # Test all major API functions
        test_functions = [
            ("get_system_overview", self.director.get_system_overview),
            ("get_learning_analysis", self.director.get_learning_analysis),
            ("analyze_system_health", self.director.analyze_system_health),
            ("get_action_effectiveness", self.director.get_action_effectiveness),
            ("get_coordinate_intelligence", self.director.get_coordinate_intelligence),
            ("get_performance_summary", self.director.get_performance_summary),
            ("get_global_counters", self.integration.get_global_counters),
            ("get_game_results", self.integration.get_game_results),
            ("get_self_model_entries", lambda: self.integration.get_self_model_entries(limit=10))
        ]
        
        for test_name, test_func in test_functions:
            total_tests += 1
            try:
                await test_func()
                tests_passed += 1
            except Exception as e:
                print(f"   ⚠️ {test_name} failed: {e}")
        
        return {
            'passed': tests_passed,
            'total': total_tests,
            'success_rate': tests_passed / total_tests * 100
        }
    
    async def _add_session_reflection(self, system_overview, learning_analysis, system_health):
        """Add current session reflection to self-model."""
        try:
            reflection_content = f"""DIRECTOR SESSION INITIALIZATION: Comprehensive system analysis completed. Current status: {system_overview.get('system_status', {})}. Learning patterns: {len(learning_analysis.get('patterns', []))} patterns. System health: {system_health.get('overall_status', 'Unknown')}. Database integration functional. Ready for autonomous operation with focus on game completion and learning optimization."""
            
            metadata = {
                "session_type": "initialization",
                "system_status": system_overview.get('system_status', {}),
                "learning_patterns": len(learning_analysis.get('patterns', [])),
                "system_health": system_health.get('overall_status', 'Unknown'),
                "timestamp": datetime.now().isoformat()
            }
            
            success = await self.integration.add_self_model_entry(
                type="reflection",
                content=reflection_content,
                session_id=self.session_id,
                importance=3,
                metadata=metadata
            )
            
            if success:
                print("   ✅ Session reflection added")
            else:
                print("   ⚠️ Failed to add session reflection")
                
        except Exception as e:
            print(f"   ⚠️ Error adding reflection: {e}")
    
    def _generate_summary(self, system_overview, learning_analysis, system_health, connectivity_results):
        """Generate comprehensive system summary."""
        status = system_overview.get('system_status', {})
        
        summary = f"""🎯 DIRECTOR STATUS: OPERATIONAL
📊 System Overview:
   • Active Sessions: {status.get('active_sessions', 0)}
   • Total Actions: {status.get('total_actions', 0)}
   • Total Wins: {status.get('total_wins', 0)}
   • Win Rate: {status.get('avg_win_rate', 0.0):.1f}%
   • Avg Score: {status.get('avg_score', 0.0):.1f}

🧠 Learning Status:
   • Patterns: {len(learning_analysis.get('patterns', []))}
   • Learning Analysis: {learning_analysis.get('summary', {})}

🏥 Health Status:
   • Overall: {system_health.get('overall_status', 'Unknown')}
   • Recommendations: {len(system_health.get('recommendations', []))}

🔍 Database Status:
   • Connectivity: {connectivity_results['success_rate']:.1f}% ({connectivity_results['passed']}/{connectivity_results['total']})
   • Integration: {'✅ Functional' if connectivity_results['success_rate'] > 80 else '⚠️ Issues detected'}

🎮 Ready for: Autonomous operation, training sessions, game completion"""
        
        return summary

async def main():
    """Main initialization function."""
    initializer = DirectorInitializer()
    success = await initializer.initialize_director()
    
    if success:
        print("\n🎯 Director ready for autonomous operation!")
        return 0
    else:
        print("\n❌ Director initialization failed!")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
