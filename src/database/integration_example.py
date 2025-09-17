"""
INTEGRATION EXAMPLE
Shows how to replace file I/O operations with database calls
"""

import asyncio
from typing import Dict, List, Any
from .system_integration import get_system_integration
from .director_commands import get_director_commands

class DatabaseIntegrationExample:
    """
    Example showing how to integrate database calls into existing systems.
    This replaces the old file-based approach with database operations.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.director = get_director_commands()
    
    # ============================================================================
    # REPLACING OLD FILE I/O WITH DATABASE CALLS
    # ============================================================================
    
    async def old_way_save_session(self, session_id: str, data: Dict[str, Any]):
        """OLD WAY: Save session data to JSON file"""
        # This is what the old system did:
        # import json
        # with open(f"data/sessions/{session_id}.json", 'w') as f:
        #     json.dump(data, f)
        pass
    
    async def new_way_save_session(self, session_id: str, data: Dict[str, Any]):
        """NEW WAY: Save session data to database"""
        return await self.integration.update_session_metrics(session_id, data)
    
    async def old_way_load_session(self, session_id: str):
        """OLD WAY: Load session data from JSON file"""
        # This is what the old system did:
        # import json
        # with open(f"data/sessions/{session_id}.json", 'r') as f:
        #     return json.load(f)
        pass
    
    async def new_way_load_session(self, session_id: str):
        """NEW WAY: Load session data from database"""
        return await self.integration.get_session_status(session_id)
    
    async def old_way_save_action_intelligence(self, game_id: str, intelligence_data: Dict[str, Any]):
        """OLD WAY: Save action intelligence to JSON file"""
        # This is what the old system did:
        # import json
        # with open(f"data/action_intelligence_{game_id}.json", 'w') as f:
        #     json.dump(intelligence_data, f)
        pass
    
    async def new_way_save_action_intelligence(self, game_id: str, intelligence_data: Dict[str, Any]):
        """NEW WAY: Save action intelligence to database"""
        return await self.integration.save_action_intelligence(game_id, intelligence_data)
    
    async def old_way_load_action_intelligence(self, game_id: str):
        """OLD WAY: Load action intelligence from JSON file"""
        # This is what the old system did:
        # import json
        # with open(f"data/action_intelligence_{game_id}.json", 'r') as f:
        #     return json.load(f)
        pass
    
    async def new_way_load_action_intelligence(self, game_id: str):
        """NEW WAY: Load action intelligence from database"""
        return await self.integration.load_action_intelligence(game_id)
    
    # ============================================================================
    # NEW CAPABILITIES WITH DATABASE
    # ============================================================================
    
    async def get_real_time_status(self):
        """Get real-time system status - NEW CAPABILITY"""
        return await self.director.get_system_overview()
    
    async def analyze_learning_progress(self, game_id: str = None):
        """Analyze learning progress - NEW CAPABILITY"""
        return await self.director.get_learning_progress(game_id)
    
    async def get_system_health(self):
        """Get system health analysis - NEW CAPABILITY"""
        return await self.director.analyze_system_health()
    
    async def get_action_effectiveness_analysis(self, game_id: str = None):
        """Get detailed action effectiveness analysis - NEW CAPABILITY"""
        return await self.director.get_action_effectiveness(game_id)
    
    async def get_coordinate_intelligence_analysis(self, game_id: str = None):
        """Get detailed coordinate intelligence analysis - NEW CAPABILITY"""
        return await self.director.get_coordinate_intelligence(game_id)
    
    async def create_new_training_session(self, mode: str = "maximum-intelligence"):
        """Create new training session - NEW CAPABILITY"""
        return await self.director.create_training_session(mode)
    
    async def log_system_event(self, level: str, component: str, message: str, **kwargs):
        """Log system event with structured data - NEW CAPABILITY"""
        return await self.integration.log_system_event(level, component, message, **kwargs)
    
    # ============================================================================
    # EXAMPLE USAGE PATTERNS
    # ============================================================================
    
    async def example_training_session_workflow(self):
        """Example of how to use the database system in a training session"""
        
        # 1. Create a new training session
        session_result = await self.create_new_training_session("maximum-intelligence")
        session_id = session_result["session_id"]
        
        print(f"🚀 Created training session: {session_id}")
        
        # 2. Log session start
        await self.log_system_event("INFO", "learning_loop", "Training session started", 
                                  session_id=session_id)
        
        # 3. Simulate some training data
        game_id = "example-game-123"
        
        # Update action effectiveness
        await self.integration.update_action_effectiveness(
            game_id, 6, 10, 3, 0.3, 2.5
        )
        
        # Update coordinate intelligence
        await self.integration.update_coordinate_intelligence(
            game_id, 32, 32, 5, 2, 0.4, 1
        )
        
        # Log game result
        await self.integration.log_game_result(game_id, session_id, {
            "status": "completed",
            "final_score": 75.0,
            "total_actions": 10,
            "win_detected": True,
            "level_completions": 1
        })
        
        print(f"📊 Logged training data for game: {game_id}")
        
        # 4. Get real-time analysis
        status = await self.get_real_time_status()
        print(f"📈 Current system status: {status['system_status']['active_sessions']} active sessions")
        
        # 5. Analyze learning progress
        learning = await self.analyze_learning_progress(game_id)
        print(f"🧠 Learning analysis: {len(learning['learning_metrics'])} metrics tracked")
        
        # 6. Check system health
        health = await self.get_system_health()
        print(f"🏥 System health: {health['status']} (score: {health['health_score']:.2f})")
        
        # 7. Get action effectiveness analysis
        actions = await self.get_action_effectiveness_analysis(game_id)
        print(f"🎯 Action analysis: {len(actions['effectiveness_data'])} actions tracked")
        
        # 8. Get coordinate intelligence analysis
        coordinates = await self.get_coordinate_intelligence_analysis(game_id)
        print(f"📍 Coordinate analysis: {len(coordinates['coordinate_data'])} coordinates tracked")
        
        # 9. Update session metrics
        await self.new_way_save_session(session_id, {
            "total_actions": 10,
            "total_wins": 1,
            "total_games": 1,
            "win_rate": 1.0,
            "avg_score": 75.0
        })
        
        print(f"💾 Updated session metrics for: {session_id}")
        
        # 10. Get final session status
        final_status = await self.new_way_load_session(session_id)
        print(f"📋 Final session status: {final_status['status'] if final_status else 'Not found'}")
        
        return session_id

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_database_integration():
    """Demonstrate the database integration capabilities"""
    
    print("🔧 DATABASE INTEGRATION DEMONSTRATION")
    print("=" * 50)
    
    # Create integration example
    example = DatabaseIntegrationExample()
    
    # Run example workflow
    session_id = await example.example_training_session_workflow()
    
    print("\n" + "=" * 50)
    print("✅ DATABASE INTEGRATION DEMONSTRATION COMPLETE")
    print("=" * 50)
    print("🎯 Key Benefits Demonstrated:")
    print("   • Real-time data access")
    print("   • Structured data storage")
    print("   • Advanced analytics capabilities")
    print("   • System health monitoring")
    print("   • Learning progress tracking")
    print("   • Action effectiveness analysis")
    print("   • Coordinate intelligence analysis")
    print()
    print("🚀 Ready to replace file I/O operations!")

if __name__ == "__main__":
    asyncio.run(demonstrate_database_integration())
