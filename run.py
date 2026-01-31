"""
NeuroQuant Platform Launcher
Starts the trading platform with proper environment setup
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment
os.environ['PYTHONPATH'] = str(project_root)

if __name__ == "__main__":
    # Import and run
    from backend.main import app, uvicorn, config
    
    print("\n" + "="*60)
    print("🚀 Starting NeuroQuant")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.api.RELOAD,
        log_level=config.logging.LEVEL.lower()
    )
