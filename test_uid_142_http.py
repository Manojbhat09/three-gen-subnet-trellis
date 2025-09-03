#!/usr/bin/env python3
"""
Simple HTTP test for UID 142
"""

import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_http():
    """Test HTTP connectivity"""
    url = "http://129.146.3.173:8092/"
    logger.info(f"🌐 Testing HTTP GET: {url}")

    try:
        response = requests.get(url, timeout=10)
        logger.info(f"✅ Status Code: {response.status_code}")
        logger.info(f"✅ Response Headers: {dict(response.headers)}")
        logger.info(f"✅ Content Length: {len(response.text)}")
        logger.info(f"✅ Content Preview: {response.text[:200]}...")

        # Check if it's a Bittensor endpoint
        if "bittensor" in response.text.lower() or "synapse" in response.text.lower():
            logger.info("✅ Looks like a Bittensor endpoint!")
        else:
            logger.warning("⚠️ Doesn't look like a Bittensor endpoint")

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ HTTP request failed: {e}")

if __name__ == "__main__":
    test_http()

