# ARC-AGI-3 API Guide for AI Agents

## Overview
ARC-AGI-3 is an interactive reasoning benchmark where agents interact with 2D grid environments through a standardized API. Agents receive visual frames (64×64 grid of color indices 0-15) and respond with actions.

## Core Actions
| Action | Description | Parameters |
|--------|-------------|------------|
| RESET | Initialize/restart game | game_id, card_id, [guid] |
| ACTION1-4 | Directional movement | game_id, guid |
| ACTION5 | Context interaction | game_id, guid |
| **ACTION6** | **Coordinate targeting** | **game_id, guid, x, y** |
| ACTION7 | Undo last action | game_id, guid |

## API Response Structure
All action responses follow this format:
```json
{
  "game_id": "ls20-016295f7601e",
  "guid": "2fa5332c-2e55-4825-b5c5-df960d504470",
  "frame": [[[0,0,1,"…"],["…"]]],
  "state": "NOT_FINISHED",
  "score": 3,
  "win_score": 254,
  "action_input": {"id": 7},
  "available_actions": [1,2,3,4,7]
}
```

## Frame Data Processing
The frame contains visual state information as a 64×64 grid:

```python
# Example frame processing
frame_data = response.json()["frame"]
current_frame = frame_data[0]  # Most recent frame

# Access specific coordinates
cell_value = current_frame[y][x]  # Value 0-15 representing color

# Full grid iteration
for y in range(64):
    for x in range(64):
        color_index = current_frame[y][x]
        # Process cell value
```

## Available Actions Handling
The `available_actions` list indicates valid next moves:

```python
available_actions = response.json()["available_actions"]

# Check specific action availability
if 1 in available_actions:
    # ACTION1 (up) is available
    pass
if 6 in available_actions:
    # ACTION6 (targeting) is available
    pass

# Action priority recommendation
def choose_action(available_actions):
    # Prefer simple actions first
    simple_actions = [1,2,3,4,5,7]
    for action in simple_actions:
        if action in available_actions:
            return action
    # Fall back to ACTION6 if no simple actions available
    if 6 in available_actions:
        return 6
    return None
```

## Critical: ACTION6 Protocol
ACTION6 is a targeted interaction command, not just movement. Think of it as a touchscreen tap at coordinates (x,y).

### Required Format:
```python
payload = {
    "game_id": "actual_game_id",
    "guid": "session_guid",
    "x": integer(0-63),  # Required
    "y": integer(0-63)   # Required
}
```

### Advanced Targeting with OpenCV:
Before using ACTION6, analyze the frame to identify interactive elements:

```python
import cv2
import numpy as np

def analyze_frame(frame_data):
    # Convert to numpy array
    grid = np.array(frame_data[0], dtype=np.uint8)
    
    # Color-based targeting (example: find red elements)
    hsv = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    
    # Find centroids of detected regions
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Minimum size threshold
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                targets.append((cx, cy))
    
    return targets
```

## Game Flow Management
1. **Open scorecard** → Get card_id
2. **RESET game** → Get guid and initial state
3. **Execute actions** (check available_actions list)
4. **Analyze responses** for state, score, and frame changes
5. **Close scorecard** when done

## State Management
- `NOT_FINISHED`: Continue playing
- `WIN`: Objective completed successfully
- `GAME_OVER`: Terminated (max actions or failure)
- `NOT_STARTED`: Requires RESET to begin

## Performance Tracking
- `score`: Current cumulative performance (0-254)
- `win_score`: Target score for victory
- Monitor score changes to evaluate action effectiveness

## Best Practices
1. Always check `available_actions` before choosing action
2. Prefer simple actions (1-5,7) when available
3. Use ACTION6 for targeted interaction with visual elements
4. Track position internally (not provided in responses)
5. Implement boundary checks (0 ≤ x,y ≤ 63)
6. Use score changes as performance feedback

## Error Handling
- Check response status codes (200 = success, 429 = rate limit)
- Validate coordinate boundaries before ACTION6
- Handle game state transitions appropriately

## Rate Limits
- 600 requests per minute
- Exponential backoff on 429 errors

This framework enables agents to evolve from blind movers to visual-interactive systems capable of sophisticated reasoning and problem-solving through targeted interactions with the game environment.