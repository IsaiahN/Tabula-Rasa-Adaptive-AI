#!/usr/bin/env python3
"""
Remove all emojis from continuous_learning_loop.py to fix Unicode issues
"""

import re

def remove_emojis(text):
    """Remove emojis and other Unicode symbols from text."""
    # Remove common emojis and symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub('', text)

def main():
    file_path = "src/arc_integration/continuous_learning_loop.py"
    
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Removing emojis...")
    cleaned_content = remove_emojis(content)
    
    # Count changes
    original_lines = content.count('\n')
    cleaned_lines = cleaned_content.count('\n')
    
    print(f"Original lines: {original_lines}")
    print(f"Cleaned lines: {cleaned_lines}")
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"✅ Removed emojis from {file_path}")

if __name__ == '__main__':
    main()
