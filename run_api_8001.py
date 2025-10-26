"""
Startup script for the Fire Door Detection API on port 8001
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check if API key is configured
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in a .env file or environment variable")
    
    print("🚀 Starting Fire Door Detection API on port 8001...")
    print("📡 API will be available at: http://localhost:8001")
    print("📚 API documentation at: http://localhost:8001/docs")
    print("🛑 Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
