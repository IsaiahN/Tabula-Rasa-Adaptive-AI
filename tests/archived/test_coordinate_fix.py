"""Simple test of coordinate generation logic."""

import random

def _get_strategic_coordinates(action: int, grid_dims: tuple) -> tuple:
    """Generate strategic default coordinates for actions with exploration."""
    grid_width, grid_height = grid_dims
    
    # Strategic coordinate selection based on action type and grid analysis
    if action == 1:  # Often drawing/placing
        # Explore upper left quadrant with some variation
        base_x, base_y = grid_width // 4, grid_height // 4
        x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
        y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
        return (x, y)
    elif action == 2:  # Often modifying  
        # Explore center area with variation
        base_x, base_y = grid_width // 2, grid_height // 2
        x = max(1, min(grid_width - 1, base_x + random.randint(-5, 5)))
        y = max(1, min(grid_height - 1, base_y + random.randint(-5, 5)))
        return (x, y)
    elif action == 3:  # Often erasing/removing
        # Explore lower right quadrant
        base_x, base_y = 3 * grid_width // 4, 3 * grid_height // 4
        x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
        y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
        return (x, y)
    elif action == 4:  # Often pattern-related
        # Explore lower left quadrant
        base_x, base_y = grid_width // 4, 3 * grid_height // 4
        x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
        y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
        return (x, y)
    elif action == 5:  # Often transformation
        # Explore upper right quadrant
        base_x, base_y = 3 * grid_width // 4, grid_height // 4
        x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
        y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
        return (x, y)
    elif action == 6:  # Special coordinate-based action - explore more broadly
        # For action 6, explore different grid regions more systematically
        regions = [
            (grid_width // 4, grid_height // 4),      # Upper left
            (3 * grid_width // 4, grid_height // 4),  # Upper right  
            (grid_width // 4, 3 * grid_height // 4),  # Lower left
            (3 * grid_width // 4, 3 * grid_height // 4), # Lower right
            (grid_width // 2, grid_height // 2),      # Center
            (grid_width // 8, grid_height // 8),      # Far upper left
            (7 * grid_width // 8, 7 * grid_height // 8), # Far lower right
        ]
        base_x, base_y = random.choice(regions)
        x = max(1, min(grid_width - 1, base_x + random.randint(-4, 4)))
        y = max(1, min(grid_height - 1, base_y + random.randint(-4, 4)))
        return (x, y)
    else:
        # For any other actions, explore the entire grid more broadly
        x = random.randint(2, grid_width - 2)
        y = random.randint(2, grid_height - 2)
        return (x, y)

def test_coordinate_variation():
    """Test that ACTION 6 coordinates are now varied instead of always (33,33)."""
    print("🧪 Testing ACTION 6 Coordinate Variation")
    print("=" * 50)
    
    # Test coordinate generation for ACTION 6 multiple times
    coordinates = []
    for i in range(20):
        x, y = _get_strategic_coordinates(6, (64, 64))
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
        print("   No more fixed (33,33) problem!")
    else:
        print("❌ PROBLEM: Coordinates are still fixed!")
    
    print(f"\n🎯 Sample of {min(10, len(unique_coords))} unique coordinates:")
    for i, coord in enumerate(sorted(unique_coords)[:10]):
        print(f"   {coord}")

if __name__ == "__main__":
    test_coordinate_variation()
