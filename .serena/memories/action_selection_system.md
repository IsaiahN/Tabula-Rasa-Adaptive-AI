# Action Selection System Analysis

## Overview
The Tabula Rasa action selection system uses a sophisticated multi-factor scoring approach rather than simple action ID priority.

## Key Components

### Action Validation and Prioritization
- Located in `src/training/analysis/action_selector.py:2993-3016`
- Removes action 0 completely
- Prioritizes actions 1-4, but this is just ordering, not final selection

### Multi-Source Suggestion System
The system generates suggestions from 17+ different sources:
1. Button-first discovery
2. Frame analysis (OpenCV)
3. Pattern matching
4. Coordinate intelligence
5. Learning-based suggestions
6. Exploration strategies
7. Advanced cognitive systems
8. Bayesian pattern detection
9. GAN-generated suggestions
10. Game-specific knowledge
11. Visual-interactive targeting
12. Strategy discovery
13. And more...

### Intelligent Selection Process
- Located in `_select_best_action_intelligent()` (lines 1449-1546)
- Uses multi-factor scoring with breakdown analysis
- Considers confidence, source reliability, learning factors, frame analysis
- Can override action priority based on confidence scores

### Action 6 Specialization
- Treats Action 6 as coordinate-based (64x64 grid)
- Has specialized button discovery and visual targeting systems
- Computer vision integration for detecting interactive elements
- Can be prioritized in "Action 6 centric games"

## Key Finding
Actions 1-4 have validation priority but NOT selection priority. The final action is chosen based on confidence scores from multiple AI systems analyzing the visual frame and past learning data.