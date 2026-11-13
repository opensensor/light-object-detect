#!/usr/bin/env python3
"""
Script to run the object detection API server.
"""

import argparse
import sys
import uvicorn
from pathlib import Path

# Resolve main:app when this script is launched by path from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the object detection API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting API server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
