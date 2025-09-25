import asyncio
import sys
from pathlib import Path


def _ensure_repo_on_path():
    # Add repository root to sys.path so 'src' package imports resolve when running
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


async def _main():
    _ensure_repo_on_path()
    try:
        from src.core.autonomous_system_manager import get_autonomy_summary, start_autonomous_system
    except Exception as e:
        print('Failed to import autonomous manager:', e)
        return

    try:
        summary = get_autonomy_summary()
        print('AUTONOMY SUMMARY:', summary)
    except Exception as e:
        print('Failed to get autonomy summary:', e)

    try:
        await start_autonomous_system('autonomous')
        print('start_autonomous_system invoked')
    except Exception as e:
        print('Failed to start autonomous system:', e)


if __name__ == '__main__':
    asyncio.run(_main())
