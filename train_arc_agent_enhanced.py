#!/usr/bin/env python3
"""
Enhanced Train ARC Agent with Coordinate-Aware System Integration
Integrates the new coordinate system with the existing training infrastructure.
"""
import asyncio
import sys
import argparse
import logging
from pathlib import Path

from arc_integration.continuous_learning_loop import ContinuousLearningLoop
from arc_integration.coordinate_aware_integration import CoordinateAwareTrainingManager
from core.salience_system import SalienceMode

logger = logging.getLogger(__name__)


class EnhancedARCTrainingManager:
    """Enhanced training manager that integrates coordinate awareness with existing system."""
    
    def __init__(self, api_key: str, arc_agents_path: str = None, use_coordinates: bool = True):
        self.api_key = api_key
        self.arc_agents_path = arc_agents_path
        self.use_coordinates = use_coordinates
        
        # Initialize both systems
        self.coordinate_manager = CoordinateAwareTrainingManager(api_key, arc_agents_path)
        self.continuous_loop = ContinuousLearningLoop(
            api_key=api_key,
            tabula_rasa_path=str(Path(__file__).parent),
            arc_agents_path=arc_agents_path
        )
    
    async def run_enhanced_training(
        self, 
        games: list, 
        mode: str = 'enhanced',
        max_actions_per_game: int = 1000,
        compare_systems: bool = False
    ):
        """
        Run enhanced training with optional system comparison.
        
        Args:
            games: List of game IDs to train on
            mode: Training mode ('enhanced', 'traditional', 'comparison')
            max_actions_per_game: Maximum actions per game session
            compare_systems: Whether to run comparison between coordinate-aware and traditional
        """
        print(f"🚀 Starting Enhanced ARC Training")
        print(f"📋 Games: {games}")
        print(f"🔧 Mode: {mode}")
        print(f"📊 Coordinate System: {'Enabled' if self.use_coordinates else 'Disabled'}")
        
        if compare_systems:
            print("🔬 Running system comparison...")
            results = await self.coordinate_manager.run_coordinate_system_benchmark(games)
            self._print_comparison_results(results)
            return results
        
        elif mode == 'enhanced' and self.use_coordinates:
            print("✨ Running coordinate-aware training...")
            results = {}
            for game_id in games:
                print(f"\n🎮 Training on {game_id}")
                game_result = await self.coordinate_manager.run_enhanced_training_session(
                    game_id=game_id,
                    max_actions=max_actions_per_game,
                    use_coordinate_awareness=True
                )
                results[game_id] = game_result
                
                # Print game summary
                self._print_game_summary(game_id, game_result)
            
            return results
            
        else:
            print("🔄 Running traditional training...")
            results = {}
            for game_id in games:
                print(f"\n🎮 Training on {game_id}")
                # Use existing continuous learning loop
                game_result = await self.continuous_loop.train_on_specific_game(
                    game_id=game_id,
                    max_mastery_sessions=10,
                    target_performance={'win_rate': 0.3, 'avg_score': 50}
                )
                results[game_id] = game_result
                
                # Print game summary
                self._print_game_summary(game_id, game_result)
            
            return results
    
    def _print_game_summary(self, game_id: str, result: dict):
        """Print summary of game training results."""
        if 'error' in result:
            print(f"❌ {game_id}: Error - {result['error']}")
            return
            
        actions_taken = result.get('actions_taken', 0)
        final_score = 0
        if 'score_progression' in result and result['score_progression']:
            final_score = result['score_progression'][-1].get('score', 0)
        elif 'final_score' in result:
            final_score = result['final_score']
            
        coord_actions = result.get('coordinate_actions', 0)
        coord_successes = result.get('coordinate_successes', 0)
        
        print(f"✅ {game_id}: {actions_taken} actions, final score: {final_score}")
        if coord_actions > 0:
            success_rate = coord_successes / coord_actions * 100
            print(f"   🎯 Coordinate actions: {coord_actions}, success rate: {success_rate:.1f}%")
    
    def _print_comparison_results(self, results: dict):
        """Print detailed comparison results."""
        print("\n" + "="*60)
        print("🔬 COORDINATE SYSTEM BENCHMARK RESULTS")
        print("="*60)
        
        games_tested = results.get('games_tested', [])
        comparisons = results.get('performance_comparison', {})
        
        for game_id in games_tested:
            if game_id in comparisons:
                comp = comparisons[game_id]
                coord_score = comp.get('coordinate_aware_final_score', 0)
                trad_score = comp.get('traditional_final_score', 0)
                improvement = comp.get('coordinate_improvement', 0)
                coord_actions = comp.get('coordinate_actions_used', 0)
                success_rate = comp.get('coordinate_success_rate', 0) * 100
                
                print(f"\n🎮 {game_id}:")
                print(f"   📊 Coordinate-aware: {coord_score} points")
                print(f"   📊 Traditional: {trad_score} points")
                print(f"   {'📈' if improvement > 0 else '📉'} Improvement: {improvement:+.1f} points")
                if coord_actions > 0:
                    print(f"   🎯 Coordinate actions used: {coord_actions} (success: {success_rate:.1f}%)")
        
        # Overall summary
        total_improvements = [comparisons[g]['coordinate_improvement'] for g in games_tested if g in comparisons]
        if total_improvements:
            avg_improvement = sum(total_improvements) / len(total_improvements)
            positive_improvements = sum(1 for imp in total_improvements if imp > 0)
            
            print(f"\n🏆 OVERALL RESULTS:")
            print(f"   Average improvement: {avg_improvement:+.1f} points")
            print(f"   Games improved: {positive_improvements}/{len(games_tested)}")
            print(f"   Success rate: {positive_improvements/len(games_tested)*100:.1f}%")


async def main():
    """Main entry point for enhanced ARC training."""
    parser = argparse.ArgumentParser(description='Enhanced ARC Agent Training with Coordinate System')
    parser.add_argument('--api-key', required=True, help='ARC-AGI-3 API key')
    parser.add_argument('--arc-agents-path', help='Path to arc-agents repository')
    parser.add_argument('--games', nargs='+', 
                       default=['sp80-5f3511b239b8', 'sp80-2c4e9f6e9f8a', 'sp80-8d2e5a6f7c9b'],
                       help='List of game IDs to train on')
    parser.add_argument('--mode', choices=['enhanced', 'traditional', 'comparison'],
                       default='enhanced', help='Training mode')
    parser.add_argument('--max-actions', type=int, default=1000,
                       help='Maximum actions per game')
    parser.add_argument('--disable-coordinates', action='store_true',
                       help='Disable coordinate-aware system')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize enhanced training manager
    use_coordinates = not args.disable_coordinates
    manager = EnhancedARCTrainingManager(
        api_key=args.api_key,
        arc_agents_path=args.arc_agents_path,
        use_coordinates=use_coordinates
    )
    
    # Run training
    try:
        results = await manager.run_enhanced_training(
            games=args.games,
            mode=args.mode,
            max_actions_per_game=args.max_actions,
            compare_systems=(args.mode == 'comparison')
        )
        
        print(f"\n✅ Enhanced training completed!")
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = asyncio.run(main())
    sys.exit(0 if results else 1)
