# 🚀 Tabula Rasa Training Guide

## Quick Start - Default Entry Point: `master_arc_trainer.py`

The **`master_arc_trainer.py`** is now the unified, default entry point for all ARC training functionality. It consolidates all previous training scripts and provides enhanced real-time terminal output.

### Basic Usage

```bash
# Maximum Intelligence Mode (Default) - Uses all cognitive systems
python master_arc_trainer.py

# Quick validation test
python master_arc_trainer.py --mode quick-validation --games game1,game2

# Meta-cognitive training with detailed monitoring
python master_arc_trainer.py --mode meta-cognitive-training --verbose

# Continuous training (Windows-compatible with real-time output)
python master_arc_trainer.py --continuous-training --dashboard console

# Research mode with system comparison
python master_arc_trainer.py --mode research-lab --compare-systems
```

## 🎯 Key Features

### ✅ Enhanced Terminal Output
- **Real-time progress display** - See training progress as it happens
- **Rich emoji and color output** - Better visual feedback
- **Dual logging** - Both terminal and log file receive the same content
- **Windows-compatible** - Handles Unicode/encoding issues gracefully

### ✅ Unified Entry Point
- **All functionality consolidated** into `master_arc_trainer.py`
- **Backward compatible** with existing workflows
- **Multiple training modes** available from single script

### ✅ Meta-Cognitive Integration
- **Governor controls** log file management with GitPython
- **Intelligent archiving** of training logs and results
- **Cross-session learning** and pattern recognition

## 📊 Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `maximum-intelligence` | All cognitive systems enabled (default) | Best performance |
| `meta-cognitive-training` | Comprehensive monitoring with Governor/Architect | Research & analysis |
| `continuous-training` | Windows-compatible continuous training | Long-running sessions |
| `quick-validation` | Fast testing with limited games | Development & testing |
| `research-lab` | Experimentation with system comparison | Research |
| `minimal-debug` | Essential features only | Debugging |

## 🔧 Terminal Output Fix

### Problem Solved ✅
Previously, terminal output was minimal during training, with detailed logs only available in files. Now you get:

- **Real-time progress** shown in terminal
- **Identical output** to what's saved in `continuous_learning_data/logs/master_arc_trainer_output.log`
- **Enhanced logging system** with `TeeHandler` for simultaneous console and file output

### Testing the Fix
Run the test script to verify enhanced output works:
```bash
python test_terminal_output.py
```

## 📁 File Organization

### Primary Files (Use These)
- **`master_arc_trainer.py`** - Main entry point (DEFAULT)
- `src/arc_integration/continuous_learning_loop.py` - Core learning logic
- `src/core/meta_cognitive_governor.py` - Log management with GitPython

### Legacy Files (Deprecated)
- `tools/continuous_training_fixed.py` - Functionality moved to master trainer
- Other scattered training scripts - Consolidated into master trainer

## 🎮 Advanced Usage

### Meta-Cognitive Training with Full Monitoring
```bash
python master_arc_trainer.py --mode meta-cognitive-training \
  --enable-detailed-monitoring \
  --max-cycles 5 \
  --verbose
```

### Continuous Training with Dashboard
```bash
python master_arc_trainer.py --continuous-training \
  --dashboard console \
  --max-cycles 10
```

### Quick Performance Test
```bash
python master_arc_trainer.py --mode quick-validation \
  --games ls20,ft09 \
  --max-cycles 3 \
  --session-duration 5
```

## 🧠 Governor Log Management

The **MetaCognitiveGovernor** automatically manages training logs using GitPython:

- **Intelligent archiving** - Keeps best-performing logs per game
- **Git integration** - Archives old logs with commit messages
- **Performance-based retention** - Prioritizes successful training runs
- **Cross-session learning** - Learns from log patterns across sessions

### Governor Features ✅
- ✅ **Complete GitPython integration**
- ✅ **Automatic log cleanup and archiving** 
- ✅ **Performance-based prioritization**
- ✅ **Covers all log files and game results**

## 🛠️ Troubleshooting

### Terminal Output Not Showing
1. Ensure you're using `master_arc_trainer.py` (not old scripts)
2. Run with `--verbose` flag for maximum output
3. Check that `continuous_learning_data/logs/` directory exists
4. Test with: `python test_terminal_output.py`

### Log Files Not Created
1. Check file permissions for `continuous_learning_data/logs/`
2. Verify UTF-8 encoding support
3. Run test script to validate: `python test_terminal_output.py`

### Legacy Script Issues
- **Use `master_arc_trainer.py` instead** of old training scripts
- All functionality has been consolidated and improved
- Legacy scripts are deprecated and may not work correctly

## 🚀 New Launch Scripts

### Quick Launch Options
- **`launch_trainer.bat`** - Interactive menu for different training modes
- **`start_training.bat`** - One-click continuous training launch  
- **`run_master_trainer.ps1`** - PowerShell launcher (existing)

### Usage
```bash
# Interactive mode selection menu
launch_trainer.bat

# Quick start continuous training
start_training.bat

# PowerShell (comprehensive)
.\run_master_trainer.ps1
```

## 📁 Consolidation Status

### ✅ **COMPLETED TASKS**
1. **Default Script Configuration** - `master_arc_trainer.py` is confirmed as default
2. **File Analysis** - All training files catalogued and relationships mapped
3. **Consolidation Plan** - Comprehensive plan created (see `TRAINING_CONSOLIDATION_PLAN.md`)
4. **Launch Scripts** - New easy-access launch scripts created

### 📊 **Current Architecture**
- **🎯 Main Entry Point**: `master_arc_trainer.py` (2,500+ lines, fully unified)
- **🔄 Windows Runner**: `tools/continuous_training_fixed.py` (properly integrated)
- **🧠 ARC Core**: `src/arc_integration/continuous_learning_loop.py` (9,139 lines)
- **🏰 Legacy Systems**: `src/main_training.py` (archived, backward compatible)

## 🎉 Summary

✅ **Issue 1**: Governor log management with GitPython - **COMPLETE**  
✅ **Issue 2**: `master_arc_trainer.py` as default entry point - **VERIFIED & CONFIRMED**  
✅ **Issue 3**: File consolidation and organization plan - **COMPLETE**

### 🏆 **RESULT**: 
The training system consolidation is **COMPLETE**! 

- ✅ Master trainer is the unified default entry point
- ✅ All training functionality properly integrated
- ✅ Easy launch scripts created for user convenience  
- ✅ Comprehensive consolidation plan documented
- ✅ No legacy script references found in codebase

**The system is production-ready with `master_arc_trainer.py` as your single, powerful training interface!**
