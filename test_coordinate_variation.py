"""Test the improved coordinate system for ACTION 6."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the coordinate optimization function
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

def test_coordinate_variation():
    """Test that ACTION 6 coordinates are now varied instead of always (33,33)."""
    print("🧪 Testing ACTION 6 Coordinate Variation")
    print("=" * 50)
    
    # Create a continuous learning loop instance
    loop = ContinuousLearningLoop()
    
    # Test coordinate generation for ACTION 6 multiple times
    coordinates = []
    for i in range(20):
        x, y = loop._optimize_coordinates_for_action(6, (64, 64))
        coordinates.append((x, y))
        print(f"Test {i+1:2d}: ACTION 6 coordinates: ({x:2d}, {y:2d})")
    
    print("\n📊 Analysis:")
    unique_coords = set(coordinates)
    print(f"   Total tests: {len(coordinates)}")
    print(f"   Unique coordinates: {len(unique_coords)}")
    print(f"   Variation rate: {len(unique_coords)/len(coordinates)*100:.1f}%")
    
    # Check if we're still getting the problematic (33,33)
    count_33_33 = coordinates.count((33, 33))
    print(f"   Times (33,33) appeared: {count_33_33}/{len(coordinates)}")
    
    if len(unique_coords) > 1:
        print("✅ SUCCESS: Coordinates are now varied!")
    else:
        print("❌ PROBLEM: Coordinates are still fixed!")
    
    print("\n🎯 Unique coordinates found:")
    for coord in sorted(unique_coords):
        print(f"   {coord}")

if __name__ == "__main__":
    test_coordinate_variation()
