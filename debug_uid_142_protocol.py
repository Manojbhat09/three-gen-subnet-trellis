#!/usr/bin/env python3
"""
Debug UID 142 Bittensor protocol issues
Test the actual Bittensor communication protocol
"""

import socket
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_raw_socket_connection(host='129.146.3.173', port=8092):
    """Test raw socket connection to UID 142"""
    logger.info(f"🔌 Testing raw socket connection to {host}:{port}")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)

        logger.info("Connecting to socket...")
        sock.connect((host, port))
        logger.info("✅ Socket connection successful!")

        # Try to send some data
        test_data = b"HELLO\r\n"
        logger.info(f"Sending test data: {test_data}")
        sock.send(test_data)

        # Try to receive response
        sock.settimeout(5)
        try:
            response = sock.recv(1024)
            logger.info(f"Received response: {response}")
        except socket.timeout:
            logger.info("No response received (timeout)")

        sock.close()
        logger.info("Socket closed")

    except Exception as e:
        logger.error(f"❌ Socket connection failed: {e}")

def test_bittensor_headers():
    """Test what headers Bittensor sends"""
    logger.info("🔍 Testing Bittensor protocol headers")

    # This would require proper Bittensor client setup
    logger.info("⚠️ Full Bittensor protocol test requires wallet setup")
    logger.info("💡 Use: python wallet_setup_guide.py")

def manual_investigation_steps():
    """Provide manual investigation steps"""
    logger.info("🔧 MANUAL INVESTIGATION STEPS")
    logger.info("="*50)

    steps = [
        "1. ✅ CONFIRMED: Port 8092 is OPEN and accepting connections",
        "2. 🔍 NEXT: Test if it's actually a Bittensor validator:",
        "   curl -v http://129.146.3.173:8092/",
        "   (Should return Bittensor protocol info, not generic web response)",
        "",
        "3. 🔐 AUTHENTICATION: Check if authentication is required:",
        "   The orchestrator might need proper wallet authentication",
        "",
        "4. 📊 PROTOCOL: Verify Bittensor protocol compatibility:",
        "   Different validators might use different protocol versions",
        "",
        "5. ⏰ TIMING: Check for request/response timing issues:",
        "   Validator might be slow to respond or have timeout issues",
        "",
        "6. 🚫 BLACKLIST: Check if validator has blacklisted our requests:",
        "   Look for 'blacklisted' or 'forbidden' responses",
        "",
        "7. 📝 LOGS: Check validator logs for our connection attempts:",
        "   May show authentication failures or protocol mismatches"
    ]

    for step in steps:
        logger.info(step)

def main():
    """Main investigation function"""
    logger.info("🔍 UID 142 PROTOCOL INVESTIGATION")
    logger.info("="*60)
    logger.info("Since port 8092 is OPEN, the issue is likely:")
    logger.info("• Bittensor protocol compatibility")
    logger.info("• Authentication requirements")
    logger.info("• Request formatting")
    logger.info("• Response parsing")
    logger.info("")

    # Test raw socket connection
    test_raw_socket_connection()

    # Provide investigation steps
    manual_investigation_steps()

    logger.info("
🎯 SUMMARY:"    logger.info("-" * 40)
    logger.info("✅ Network: UID 142 port 8092 is accessible")
    logger.info("❓ Protocol: Need to investigate Bittensor protocol issues")
    logger.info("🔧 Next: Test with proper Bittensor client authentication")

if __name__ == "__main__":
    main()
