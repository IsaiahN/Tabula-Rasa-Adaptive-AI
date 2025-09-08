import os
# Load .env naive parser
try:
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line.startswith('ARC_API_KEY='):
                os.environ['ARC_API_KEY'] = line.split('=',1)[1]
                break
except Exception:
    pass
# Run the existing check
import asyncio
from tmp_api_check import main
asyncio.run(main())
