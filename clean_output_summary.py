"""Summary of Clean Output Modifications

The continuous learning loop has been modified to show only available_actions changes instead of verbose output.

CHANGES MADE:

1. In _send_enhanced_action():
   - REMOVED: Verbose "ACTION X EXECUTED → State: Y, Score: Z" messages
   - REMOVED: "New Available Actions: [...]" for every action
   - ADDED: Only show "Available Actions Changed for {game_id}: [...]" when actions actually change
   - ADDED: Tracks last known available_actions per game to detect changes

2. In investigate_api_available_actions():
   - REMOVED: "API AVAILABLE ACTIONS for {game_id}: [...]" every time
   - REMOVED: "SELECTED ACTION: X (from available: [...])" verbose logging
   - ADDED: Only show "Available Actions for {game_id}: [...]" when actions change from initial state

3. In game session management:
   - SIMPLIFIED: "LEVEL RESET for {game_id}" instead of "Attempting LEVEL RESET for {game_id} with GUID {guid}"
   - SIMPLIFIED: "NEW GAME RESET for {game_id}" instead of verbose version with scorecard details
   - SIMPLIFIED: "Scorecard opened: {id}" instead of "Opened scorecard: {id}"
   - SIMPLIFIED: Success messages without showing internal GUIDs

4. Error messages:
   - SIMPLIFIED: "ACTION X FAILED: status_code" instead of full error text dump

EXPECTED BEHAVIOR:
- When you run training, you'll see:
  🎮 Available Actions for game_123: [1, 2, 3]
  (only when actions list changes)
  
  🎮 Available Actions Changed for game_123: [1, 2, 6]
  (when actions change after an API call)
  
- No more spam of every single action execution
- Clean focus on what actions are actually available from the API
- Changes tracking so you only see updates when something meaningful happens

TESTING:
The system is ready to run with:
python train_arc_agent.py --run-mode continuous --continuous-mode direct_control_swarm

This will show only the available_actions from API responses when they change,
giving you clear visibility into what the API is returning without noise.
"""

print(__doc__)
