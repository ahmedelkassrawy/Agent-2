#!/usr/bin/env python3
"""
LiteLLM Proxy Setup for Agent Two
This script sets up and runs a LiteLLM proxy server that Agent Two can use.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def setup_environment():
    """Set up required environment variables"""
    
    required_keys = {
        'GROQ_API_KEY': 'your_groq_api_key_here',
        'GOOGLE_API_KEY': 'your_google_api_key_here',
        'LANGFUSE_PUBLIC_KEY': 'your_langfuse_public_key_here',
        'LANGFUSE_SECRET_KEY': 'your_langfuse_secret_key_here'
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            if key != 'GOOGLE_API_KEY': 
                missing_keys.append(f"{key}: {description}")
    
    if missing_keys:
        print("⚠️  Missing environment variables:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nSet these before running the proxy:")
        print("export GROQ_API_KEY='your_groq_key_here'")
        print("export LANGFUSE_PUBLIC_KEY='your_langfuse_public_key' (optional)")
        print("export LANGFUSE_SECRET_KEY='your_langfuse_secret_key' (optional)")
        return False
    
    return True

def install_dependencies():
    """Install required packages"""
    try:
        import litellm
        print("✅ LiteLLM already installed")
    except ImportError:
        print("📦 Installing LiteLLM...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "litellm[proxy]"])
    
    try:
        import langfuse
        print("✅ Langfuse already installed")
    except ImportError:
        print("📦 Installing Langfuse...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "langfuse"])

def start_proxy_server():
    """Start the LiteLLM proxy server"""
    config_path = Path(__file__).parent / "logs" / "llm.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    print(f"🚀 Starting LiteLLM proxy with config: {config_path}")
    print("📡 Proxy will be available at: http://localhost:4000")
    print("📚 Admin UI will be available at: http://localhost:4000/ui")
    print("\n🔄 Starting server...")
    
    try:
        # Start the proxy server
        cmd = [
            "litellm", 
            "--config", str(config_path),
            "--port", "4000",
            "--num_workers", "1"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️  Proxy server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting proxy: {e}")
        return False
    
    return True

def test_proxy():
    """Test the proxy server"""
    import requests
    import json
    
    try:
        # Test if proxy is running
        response = requests.get("http://localhost:4000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Proxy is healthy")
            
            # Test a model call
            test_payload = {
                "model": "groq-gemma9b",
                "messages": [{"role": "user", "content": "Hello, test message"}],
                "max_tokens": 50
            }
            
            chat_response = requests.post(
                "http://localhost:4000/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=30
            )
            
            if chat_response.status_code == 200:
                print("✅ Model call successful")
                return True
            else:
                print(f"❌ Model call failed: {chat_response.status_code}")
                return False
        else:
            print(f"❌ Proxy health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to proxy: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 Setting up LiteLLM Proxy for Agent Two")
    print("=" * 50)
    
    # Check environment
    if not setup_environment():
        print("\n❌ Environment setup failed. Please set required API keys.")
        return
    
    # Install dependencies
    print("\n📦 Checking dependencies...")
    install_dependencies()
    
    # Start proxy
    print("\n🚀 Starting proxy server...")
    start_proxy_server()

if __name__ == "__main__":
    main()
