# Test Suite Organization

This directory contains the organized test suite for the Adaptive Learning Agent project.

## Directory Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions  
├── system/         # System-level tests for full agent functionality
└── agi_puzzle_results/  # Results from AGI puzzle evaluations
```

## Test Categories

### Unit Tests (`tests/unit/`)
- `test_basic_functionality.py` - Core component validation
- `test_energy_system.py` - Energy system functionality
- `test_learning_progress.py` - Learning progress drive
- `test_memory_dnc.py` - DNC memory system
- `test_meta_learning_simple.py` - Meta-learning system basics

### Integration Tests (`tests/integration/`)
- `test_agent_on_puzzles.py` - Agent performance on AGI puzzles
- `test_agi_puzzles.py` - AGI puzzle environment validation
### System Tests (`tests/system/`)
- Available system tests vary - check the `tests/system/` directory for current files

## Running Tests

### Individual Test Files
```bash
# Unit tests
python -m pytest tests/unit/test_meta_learning_simple.py -v

# Integration tests
python -m pytest tests/integration/ -v

# System tests
python -m pytest tests/system/ -v
```

### Test Categories
```bash
# All unit tests
python -m pytest tests/unit/ -v

# All integration tests
python -m pytest tests/integration/ -v

# All system tests
python -m pytest tests/system/ -v

# All tests
python -m pytest tests/ -v
```

### Direct Execution
```bash
# From project root
python tests/unit/test_meta_learning_simple.py
# Check tests/integration/ and tests/system/ directories for available test files
```

## Test Requirements

All tests require:
- PyTorch
- NumPy
- Standard Python libraries

Some integration tests may require additional dependencies as specified in `requirements.txt`.

## Recent Enhancements

- **Meta-Learning Integration**: Tests verify meta-learning system functionality
- **Enhanced Sleep System**: Tests automatic object encoding during sleep phases
- **Proper Path Resolution**: All tests use correct relative paths for imports
- **Organized Structure**: Tests categorized by scope and complexity
