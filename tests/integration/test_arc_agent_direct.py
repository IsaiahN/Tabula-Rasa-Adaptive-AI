#!/usr/bin/env python3
"""
Direct test of ARC-AGI-3 agent to diagnose hanging issue
"""
import asyncio
import pytest
import subprocess
import sys
import time
from pathlib import Path

@pytest.mark.asyncio
async def test_arc_agent():
    """Test the ARC-AGI-3 agent directly"""
    
    arc_agents_path = Path("C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents")
    
    if not arc_agents_path.exists():
        print(f"❌ ARC-AGI-3-Agents not found at: {arc_agents_path}")
        return
    
    print(f"🔧 Testing ARC-AGI-3 agent at: {arc_agents_path}")
    
    # Test 1: Check if uv is available
    try:
        result = subprocess.run(['uv', '--version'], 
                              capture_output=True, text=True, timeout=10)
        print(f"✅ UV Version: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ UV not available: {e}")
        return
    
    # Test 2: Try to run the agent with a short timeout
    cmd = ['uv', 'run', 'main.py', '--agent=adaptivelearning', '--game=sp80-5f3511b239b8']
    
    print(f"🎯 Testing command: {' '.join(cmd)}")
    print(f"📂 Working directory: {arc_agents_path}")
    
    try:
        # Use asyncio subprocess with timeout
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(arc_agents_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        print("🔄 Subprocess started, waiting for output...")
        
        try:
            # Wait with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=15.0  # 15 second timeout
            )
            
            print(f"✅ Process completed!")
            print(f"📤 STDOUT ({len(stdout)} bytes):")
            print(stdout.decode('utf-8', errors='ignore')[:500])
            
            if stderr:
                print(f"⚠️ STDERR ({len(stderr)} bytes):")
                print(stderr.decode('utf-8', errors='ignore')[:500])
                
        except asyncio.TimeoutError:
            print(f"⏰ TIMEOUT after 15 seconds")
            print(f"🔪 Killing process...")
            process.kill()
            await process.wait()
            print(f"❌ Process killed due to timeout - this indicates the hanging issue")
            
    except Exception as e:
        print(f"❌ Subprocess error: {e}")

if __name__ == "__main__":
    print("🚀 DIRECT ARC-AGI-3 AGENT TEST")
    print("="*50)
    
    asyncio.run(test_arc_agent())
