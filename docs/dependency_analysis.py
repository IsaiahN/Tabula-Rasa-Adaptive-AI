"""
DEPENDENCY USAGE ANALYSIS - Where each package is used in the enhanced continuous learning system
"""

DEPENDENCY_USAGE = {
    'numpy': [
        "Mathematical operations for agent states",
        "Grid dimension calculations", 
        "Performance metrics computation",
        "Statistical analysis of learning progress",
        "Memory consolidation calculations"
    ],
    
    'torch': [
        "Neural network operations in predictive core",
        "Sleep system memory consolidation", 
        "Salience-weighted experience replay",
        "Memory strengthening/decay operations",
        "Gradient computations during sleep cycles"
    ],
    
    'aiohttp': [
        "Async HTTP requests to ARC-AGI-3 API",
        "Concurrent game execution in SWARM mode",
        "API response handling and parsing",
        "Timeout management for game episodes"
    ],
    
    'asyncio': [
        "SWARM mode concurrent training",
        "Sleep cycle execution without blocking",
        "Parallel game processing",
        "Async episode execution"
    ],
    
    'python-dotenv': [
        "Loading ARC_API_KEY from environment",
        "Configuration management",
        "Secure API key handling"
    ],
    
    'scipy': [
        "Advanced mathematical functions for salience",
        "Statistical distributions for memory operations",
        "Optimization routines for consolidation"
    ],
    
    # BUILT-IN PACKAGES (no installation needed)
    'json': ["API response parsing", "Results serialization"],
    'pathlib': ["File system operations", "Directory management"],
    'logging': ["System status logging", "Debug information"],
    'time': ["Performance timing", "Sleep cycle timing"],
    'concurrent.futures': ["ThreadPoolExecutor for SWARM mode"],
    'dataclasses': ["Data structure definitions"],
    'typing': ["Type annotations for better code quality"]
}

# ACTUAL IMPORT COUNT BY CRITICALITY
CRITICAL_EXTERNAL_DEPS = 4  # numpy, torch, aiohttp, python-dotenv
RECOMMENDED_DEPS = 2        # scipy, pyyaml  
BUILT_IN_DEPS = 8          # All others are Python built-ins
TOTAL_IMPORTS = 14

print("📊 DEPENDENCY BREAKDOWN:")
print(f"Critical external dependencies: {CRITICAL_EXTERNAL_DEPS}")
print(f"Recommended dependencies: {RECOMMENDED_DEPS}")  
print(f"Built-in Python modules: {BUILT_IN_DEPS}")
print(f"Total imports: {TOTAL_IMPORTS}")
print(f"\nMinimal installation: Only {CRITICAL_EXTERNAL_DEPS} packages needed!")
