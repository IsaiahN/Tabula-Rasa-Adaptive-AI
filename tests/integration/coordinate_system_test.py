"""
Test Harness for ARC-AGI-3 Coordinate System and ACTION6 Implementation
Validates proper API usage before full training integration.
"""
import sys
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.enhanced_client import ArcAgiApiClient, CoordinateManager
from vision.frame_analyzer import FrameAnalyzer
from learning.pathway_system import PathwayLearningSystem


class CoordinateSystemTester:
    """
    Test harness for validating coordinate system and ACTION6 implementation.
    """
    
    def __init__(self, api_key: str):
        self.api_client = ArcAgiApiClient(api_key)
        self.frame_analyzer = FrameAnalyzer()
        self.pathway_system = PathwayLearningSystem()
        
        self.test_results = []
        self.test_game_id = None
        self.test_scorecard_id = None
    
    def run_full_test_suite(self, game_id: str, scorecard_id: str) -> Dict[str, Any]:
        """
        Run complete test suite for coordinate system validation.
        """
        print("🧪 STARTING COORDINATE SYSTEM TEST SUITE")
        print("=" * 50)
        
        self.test_game_id = game_id
        self.test_scorecard_id = scorecard_id
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'scorecard_id': scorecard_id,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': []
        }
        
        # Test 1: Basic API Connection
        result1 = self.test_api_connection()
        results['test_details'].append(result1)
        if result1['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 2: Game Start/Reset
        result2 = self.test_game_start()
        results['test_details'].append(result2)
        if result2['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            return results  # Can't continue without game session
        
        # Test 3: Coordinate Validation
        result3 = self.test_coordinate_validation()
        results['test_details'].append(result3)
        if result3['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 4: ACTION6 Center Start Strategy
        result4 = self.test_action6_center_start()
        results['test_details'].append(result4)
        if result4['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 5: Simple Actions (ACTION1-5)
        result5 = self.test_simple_actions()
        results['test_details'].append(result5)
        if result5['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 6: Frame Analysis
        result6 = self.test_frame_analysis()
        results['test_details'].append(result6)
        if result6['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 7: Pathway Learning Integration
        result7 = self.test_pathway_learning()
        results['test_details'].append(result7)
        if result7['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        # Test 8: Score-Based Success Measurement
        result8 = self.test_score_measurement()
        results['test_details'].append(result8)
        if result8['passed']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
        
        self._print_test_summary(results)
        return results
    
    def test_api_connection(self) -> Dict[str, Any]:
        """
        Test basic API connectivity.
        """
        print("\n🔌 Test 1: API Connection")
        
        try:
            # Test if we can reach the API
            state = self.api_client.get_current_state()
            
            result = {
                'test_name': 'API Connection',
                'passed': True,
                'message': 'API client initialized successfully',
                'details': {'client_state': state}
            }
            print("   ✅ PASSED: API client ready")
            
        except Exception as e:
            result = {
                'test_name': 'API Connection',
                'passed': False,
                'message': f'API connection failed: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def test_game_start(self) -> Dict[str, Any]:
        """
        Test game initialization.
        """
        print("\n🎮 Test 2: Game Start/Reset")
        
        try:
            # Start new game
            response = self.api_client.start_game(self.test_game_id, self.test_scorecard_id)
            
            if 'error' in response:
                result = {
                    'test_name': 'Game Start',
                    'passed': False,
                    'message': f'Game start failed: {response["error"]}',
                    'details': response
                }
                print(f"   ❌ FAILED: {response['error']}")
                return result
            
            # Validate response structure
            required_fields = ['game_id', 'guid', 'frame', 'state', 'score', 'win_score', 'available_actions']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                result = {
                    'test_name': 'Game Start',
                    'passed': False,
                    'message': f'Response missing required fields: {missing_fields}',
                    'details': response
                }
                print(f"   ❌ FAILED: Missing fields {missing_fields}")
                return result
            
            result = {
                'test_name': 'Game Start',
                'passed': True,
                'message': f'Game started successfully (GUID: {response.get("guid", "N/A")})',
                'details': {
                    'game_id': response.get('game_id'),
                    'initial_score': response.get('score'),
                    'win_score': response.get('win_score'),
                    'available_actions': response.get('available_actions'),
                    'state': response.get('state')
                }
            }
            print(f"   ✅ PASSED: Game started, Score: {response.get('score', 0)}/{response.get('win_score', 0)}")
            
        except Exception as e:
            result = {
                'test_name': 'Game Start',
                'passed': False,
                'message': f'Game start exception: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def test_coordinate_validation(self) -> Dict[str, Any]:
        """
        Test coordinate validation and bounds checking.
        """
        print("\n📍 Test 3: Coordinate Validation")
        
        try:
            test_cases = [
                # Valid coordinates
                {'coords': (0, 0), 'should_pass': True, 'description': 'top-left corner'},
                {'coords': (63, 63), 'should_pass': True, 'description': 'bottom-right corner'},
                {'coords': (32, 32), 'should_pass': True, 'description': 'center'},
                
                # Invalid coordinates (should be clamped)
                {'coords': (-1, 5), 'should_pass': False, 'description': 'negative x'},
                {'coords': (5, -1), 'should_pass': False, 'description': 'negative y'},
                {'coords': (64, 32), 'should_pass': False, 'description': 'x too high'},
                {'coords': (32, 64), 'should_pass': False, 'description': 'y too high'},
            ]
            
            validation_results = []
            
            for case in test_cases:
                x, y = case['coords']
                clamped_x, clamped_y = CoordinateManager.clamp_coordinates(x, y)
                
                # Test coordinate clamping
                is_valid = (0 <= clamped_x <= 63 and 0 <= clamped_y <= 63)
                
                validation_results.append({
                    'original': (x, y),
                    'clamped': (clamped_x, clamped_y),
                    'description': case['description'],
                    'valid_after_clamp': is_valid,
                    'expected_behavior': case['should_pass']
                })
            
            # All coordinates should be valid after clamping
            all_valid = all(result['valid_after_clamp'] for result in validation_results)
            
            result = {
                'test_name': 'Coordinate Validation',
                'passed': all_valid,
                'message': 'All coordinates valid after clamping' if all_valid else 'Some coordinates invalid after clamping',
                'details': {'test_cases': validation_results}
            }
            
            if all_valid:
                print("   ✅ PASSED: All coordinate validation tests passed")
            else:
                print("   ❌ FAILED: Some coordinate validation tests failed")
            
        except Exception as e:
            result = {
                'test_name': 'Coordinate Validation',
                'passed': False,
                'message': f'Coordinate validation exception: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def test_action6_center_start(self) -> Dict[str, Any]:
        """
        Test ACTION6 with center-start strategy.
        """
        print("\n🎯 Test 4: ACTION6 Center Start Strategy")
        
        try:
            # Check if ACTION6 is available
            available_actions = self.api_client.get_available_actions()
            
            if 6 not in available_actions:
                result = {
                    'test_name': 'ACTION6 Center Start',
                    'passed': False,
                    'message': 'ACTION6 not available in current game state',
                    'details': {'available_actions': available_actions}
                }
                print(f"   ⚠️ SKIPPED: ACTION6 not available (available: {available_actions})")
                return result
            
            # Get initial state
            initial_score = self.api_client.current_score
            
            # Execute ACTION6 with center coordinates
            center_x, center_y = CoordinateManager.get_center_coordinates()
            response = self.api_client.execute_action(6, (center_x, center_y))
            
            if 'error' in response:
                result = {
                    'test_name': 'ACTION6 Center Start',
                    'passed': False,
                    'message': f'ACTION6 execution failed: {response["error"]}',
                    'details': response
                }
                print(f"   ❌ FAILED: {response['error']}")
                return result
            
            # Validate response
            new_score = response.get('score', initial_score)
            action_input = response.get('action_input', {})
            
            # Check if action was recorded correctly
            action_data = action_input.get('data', {})
            recorded_x = action_data.get('x')
            recorded_y = action_data.get('y')
            
            coordinate_match = (recorded_x == center_x and recorded_y == center_y)
            
            result = {
                'test_name': 'ACTION6 Center Start',
                'passed': coordinate_match,
                'message': f'ACTION6 executed with coordinates ({center_x}, {center_y}), recorded as ({recorded_x}, {recorded_y})',
                'details': {
                    'sent_coordinates': (center_x, center_y),
                    'recorded_coordinates': (recorded_x, recorded_y),
                    'coordinate_match': coordinate_match,
                    'score_before': initial_score,
                    'score_after': new_score,
                    'score_change': new_score - initial_score,
                    'action_input': action_input
                }
            }
            
            if coordinate_match:
                print(f"   ✅ PASSED: ACTION6 executed correctly at ({center_x}, {center_y})")
                print(f"   📊 Score: {initial_score} → {new_score} (change: {new_score - initial_score})")
            else:
                print(f"   ❌ FAILED: Coordinate mismatch - sent ({center_x}, {center_y}), recorded ({recorded_x}, {recorded_y})")
            
        except Exception as e:
            result = {
                'test_name': 'ACTION6 Center Start',
                'passed': False,
                'message': f'ACTION6 test exception: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def test_simple_actions(self) -> Dict[str, Any]:
        """
        Test simple actions (ACTION1-5, ACTION7).
        """
        print("\n🕹️ Test 5: Simple Actions (ACTION1-5, ACTION7)")
        
        try:
            available_actions = self.api_client.get_available_actions()
            simple_actions = [a for a in available_actions if a != 6]  # Exclude ACTION6
            
            if not simple_actions:
                result = {
                    'test_name': 'Simple Actions',
                    'passed': False,
                    'message': 'No simple actions available for testing',
                    'details': {'available_actions': available_actions}
                }
                print("   ⚠️ SKIPPED: No simple actions available")
                return result
            
            action_results = []
            
            # Test up to 3 simple actions
            for action in simple_actions[:3]:
                initial_score = self.api_client.current_score
                
                response = self.api_client.execute_action(action)
                
                if 'error' in response:
                    action_results.append({
                        'action': action,
                        'success': False,
                        'error': response['error']
                    })
                else:
                    new_score = response.get('score', initial_score)
                    action_results.append({
                        'action': action,
                        'success': True,
                        'score_before': initial_score,
                        'score_after': new_score,
                        'score_change': new_score - initial_score,
                        'state': response.get('state', 'unknown')
                    })
            
            successful_actions = [r for r in action_results if r['success']]
            
            result = {
                'test_name': 'Simple Actions',
                'passed': len(successful_actions) > 0,
                'message': f'{len(successful_actions)}/{len(action_results)} simple actions executed successfully',
                'details': {'action_results': action_results}
            }
            
            if len(successful_actions) > 0:
                print(f"   ✅ PASSED: {len(successful_actions)} simple actions executed")
                for action_result in successful_actions:
                    if action_result['success']:
                        print(f"   📊 ACTION{action_result['action']}: {action_result['score_before']} → {action_result['score_after']} (Δ{action_result['score_change']})")
            else:
                print("   ❌ FAILED: No simple actions executed successfully")
            
        except Exception as e:
            result = {
                'test_name': 'Simple Actions',
                'passed': False,
                'message': f'Simple actions test exception: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def test_frame_analysis(self) -> Dict[str, Any]:
        """
        Test frame analysis and computer vision system.
        """
        print("\n👁️ Test 6: Frame Analysis")
        
        try:
            # Get current frame
            frame = self.api_client.get_current_frame()
            
            if not frame:
                result = {
                    'test_name': 'Frame Analysis',
                    'passed': False,
                    'message': 'No frame data available for analysis',
                    'details': {}
                }
                print("   ❌ FAILED: No frame data available")
                return result
            
            # Analyze frame
            analysis = self.frame_analyzer.analyze_frame(frame, self.test_game_id)
            
            # Validate analysis structure
            required_fields = ['colors_detected', 'frame_changes', 'timestamp']
            has_required_fields = all(field in analysis for field in required_fields)
            
            # Check if we can extract meaningful data
            colors_found = len(analysis.get('colors_detected', set())) > 0
            
            result = {
                'test_name': 'Frame Analysis',
                'passed': has_required_fields and colors_found,
                'message': f'Frame analysis {"successful" if has_required_fields and colors_found else "failed"}',
                'details': {
                    'frame_dimensions': f"{len(frame)}x{len(frame[0])}x{len(frame[0][0])}",
                    'colors_detected': list(analysis.get('colors_detected', set())),
                    'agent_position': analysis.get('agent_position'),
                    'position_confidence': analysis.get('position_confidence', 0.0),
                    'analysis_fields': list(analysis.keys())
                }
            }
            
            if has_required_fields and colors_found:
                print(f"   ✅ PASSED: Frame analysis working, {len(analysis.get('colors_detected', []))} colors detected")
                if analysis.get('agent_position'):
                    print(f"   📍 Agent position detected: {analysis['agent_position']} (confidence: {analysis.get('position_confidence', 0):.2f})")
            else:
                print("   ❌ FAILED: Frame analysis incomplete or no colors detected")
            
        except Exception as e:
            result = {
                'test_name': 'Frame Analysis',
                'passed': False,
                'message': f'Frame analysis exception: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def test_pathway_learning(self) -> Dict[str, Any]:
        """
        Test pathway learning system integration.
        """
        print("\n🧠 Test 7: Pathway Learning Integration")
        
        try:
            # Get recent actions from API client
            action_history = self.api_client.get_action_history()
            
            if not action_history:
                result = {
                    'test_name': 'Pathway Learning',
                    'passed': False,
                    'message': 'No action history available for pathway analysis',
                    'details': {}
                }
                print("   ⚠️ SKIPPED: No action history for pathway learning")
                return result
            
            # Track actions in pathway system
            pathway_analyses = []
            
            for action_record in action_history:
                analysis = self.pathway_system.track_action(
                    action_record['action'],
                    action_record.get('coordinates', {}),
                    action_record['score_before'],
                    action_record['score_after'],
                    self.api_client.win_score,
                    self.test_game_id
                )
                pathway_analyses.append(analysis)
            
            # Get recommendations
            available_actions = self.api_client.get_available_actions()
            recommendations = self.pathway_system.get_pathway_recommendations(
                available_actions, self.api_client.current_score, self.test_game_id
            )
            
            has_recommendations = len(recommendations.get('action_weights', {})) > 0
            has_reasoning = len(recommendations.get('reasoning', [])) > 0
            
            result = {
                'test_name': 'Pathway Learning',
                'passed': has_recommendations and has_reasoning,
                'message': f'Pathway learning {"working" if has_recommendations and has_reasoning else "incomplete"}',
                'details': {
                    'actions_tracked': len(pathway_analyses),
                    'pathway_analyses': pathway_analyses,
                    'recommendations': recommendations,
                    'confidence': recommendations.get('confidence', 0.0)
                }
            }
            
            if has_recommendations and has_reasoning:
                print(f"   ✅ PASSED: Pathway learning active, {len(pathway_analyses)} actions tracked")
                print(f"   🎯 Confidence: {recommendations.get('confidence', 0.0):.2f}")
                if recommendations.get('reasoning'):
                    print(f"   💭 Top reasoning: {recommendations['reasoning'][0]}")
            else:
                print("   ❌ FAILED: Pathway learning not generating complete recommendations")
            
        except Exception as e:
            result = {
                'test_name': 'Pathway Learning',
                'passed': False,
                'message': f'Pathway learning exception: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def test_score_measurement(self) -> Dict[str, Any]:
        """
        Test score-based success measurement.
        """
        print("\n📊 Test 8: Score-Based Success Measurement")
        
        try:
            # Get score progress info
            progress_info = self.api_client.get_score_progress()
            
            # Validate score tracking
            has_valid_scores = (
                'current_score' in progress_info and
                'win_score' in progress_info and
                'progress_ratio' in progress_info
            )
            
            # Calculate some metrics
            action_history = self.api_client.get_action_history()
            if action_history:
                score_improvements = [
                    action['score_after'] - action['score_before'] 
                    for action in action_history
                ]
                total_improvement = sum(score_improvements)
                positive_improvements = [imp for imp in score_improvements if imp > 0]
                success_rate = len(positive_improvements) / len(score_improvements) if score_improvements else 0
            else:
                total_improvement = 0
                success_rate = 0
            
            result = {
                'test_name': 'Score-Based Success Measurement',
                'passed': has_valid_scores,
                'message': f'Score measurement {"working" if has_valid_scores else "failed"}',
                'details': {
                    'progress_info': progress_info,
                    'total_score_improvement': total_improvement,
                    'action_success_rate': success_rate,
                    'actions_analyzed': len(action_history) if action_history else 0
                }
            }
            
            if has_valid_scores:
                print(f"   ✅ PASSED: Score measurement working")
                print(f"   📈 Progress: {progress_info['current_score']}/{progress_info['win_score']} ({progress_info['progress_ratio']:.1%})")
                print(f"   🎯 Success rate: {success_rate:.1%} ({len(action_history) if action_history else 0} actions)")
            else:
                print("   ❌ FAILED: Score measurement not working properly")
            
        except Exception as e:
            result = {
                'test_name': 'Score-Based Success Measurement',
                'passed': False,
                'message': f'Score measurement exception: {str(e)}',
                'details': {'error': str(e)}
            }
            print(f"   ❌ FAILED: {str(e)}")
        
        return result
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """
        Print summary of all test results.
        """
        print("\n" + "=" * 50)
        print("🧪 TEST SUITE SUMMARY")
        print("=" * 50)
        
        total_tests = results['tests_passed'] + results['tests_failed']
        pass_rate = results['tests_passed'] / total_tests if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {results['tests_passed']} ✅")
        print(f"Failed: {results['tests_failed']} ❌")
        print(f"Pass Rate: {pass_rate:.1%}")
        
        if results['tests_failed'] > 0:
            print("\n⚠️ FAILED TESTS:")
            for test in results['test_details']:
                if not test['passed']:
                    print(f"   • {test['test_name']}: {test['message']}")
        
        if pass_rate >= 0.8:
            print("\n🎉 COORDINATE SYSTEM READY FOR TRAINING!")
        elif pass_rate >= 0.6:
            print("\n⚠️ COORDINATE SYSTEM MOSTLY WORKING - Consider fixing failed tests")
        else:
            print("\n❌ COORDINATE SYSTEM NEEDS WORK - Fix critical issues before training")


def main():
    """
    Run the test harness.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ARC-AGI-3 Coordinate System')
    parser.add_argument('--api-key', required=True, help='ARC-AGI-3 API key')
    parser.add_argument('--game-id', required=True, help='Game ID to test with')
    parser.add_argument('--scorecard-id', required=True, help='Scorecard ID for tracking')
    parser.add_argument('--output-file', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Run tests
    tester = CoordinateSystemTester(args.api_key)
    results = tester.run_full_test_suite(args.game_id, args.scorecard_id)
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to {args.output_file}")
    
    return results


if __name__ == "__main__":
    main()
