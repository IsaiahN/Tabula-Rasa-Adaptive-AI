#!/usr/bin/env python3
"""
Simple API connection test for ARC-AGI-3
"""

import os
import sys
import requests
import json

def test_api_connection():
    """Test connection to ARC-AGI-3 API."""
    print("🔍 Testing ARC-AGI-3 API connection...")
    
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️ python-dotenv not installed, using environment variables only")
    except Exception as e:
        print(f"⚠️ Could not load .env file: {e}")
    
    # Get API key from environment
    api_key = os.getenv('ARC_API_KEY')
    if not api_key:
        print("❌ ARC_API_KEY not found in environment or .env file")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Test API endpoint
    try:
        url = "https://three.arcprize.org/api/games"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print("🌐 Connecting to ARC-AGI-3 servers...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ API connection successful!")
            print("🚀 Ready to start training!")
            return True
        else:
            print(f"❌ API connection failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ API connection timeout - check your internet connection")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ API connection error - check your internet connection")
        return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False

if __name__ == "__main__":
    success = test_api_connection()
    sys.exit(0 if success else 1)
