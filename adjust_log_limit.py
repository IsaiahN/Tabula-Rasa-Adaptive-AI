#!/usr/bin/env python3
"""
Utility script to adjust log file line limits
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Adjust log file line limits')
    parser.add_argument('--max-lines', type=int, default=100000, 
                       help='Maximum lines per log file (default: 100000)')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show current log file statistics')
    parser.add_argument('--rotate-now', action='store_true',
                       help='Rotate logs immediately with current limit')
    
    args = parser.parse_args()
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        from core.meta_cognitive_governor import MetaCognitiveGovernor
        from config.log_config import LogConfig
        
        print("🔧 Log Limit Adjustment Utility")
        print("=" * 40)
        
        # Show current stats if requested
        if args.show_stats:
            governor = MetaCognitiveGovernor()
            stats = governor.get_log_stats()
            print("📊 Current Log Statistics:")
            for log_name, log_info in stats.items():
                if isinstance(log_info, dict) and 'lines' in log_info:
                    print(f"  {log_name}: {log_info['lines']:,} lines, {log_info['size_mb']:.1f} MB")
                else:
                    print(f"  {log_name}: {log_info}")
            print()
        
        # Set new line limit
        config = LogConfig()
        old_limit = config.get_max_lines()
        config.set_max_lines(args.max_lines)
        new_limit = config.get_max_lines()
        
        print(f"📝 Line limit changed: {old_limit:,} → {new_limit:,}")
        
        # Rotate logs if requested
        if args.rotate_now:
            governor = MetaCognitiveGovernor()
            result = governor.rotate_logs()
            if result.get('success'):
                print("✅ Logs rotated successfully")
                if 'stats' in result:
                    print("📊 Post-rotation stats:")
                    for log_name, log_info in result['stats'].items():
                        if isinstance(log_info, dict) and 'lines' in log_info:
                            print(f"  {log_name}: {log_info['lines']:,} lines, {log_info['size_mb']:.1f} MB")
            else:
                print(f"❌ Log rotation failed: {result.get('error', 'Unknown error')}")
        
        print("✅ Configuration updated successfully!")
        print(f"💡 The new limit ({new_limit:,} lines) will be used for future log rotations.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
