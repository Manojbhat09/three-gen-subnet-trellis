#!/usr/bin/env python3
"""
Simple network connectivity test for UID 142
Tests basic TCP connectivity without Bittensor wallet requirements
"""

import socket
import time
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_tcp_connectivity(host, port, timeout=10):
    """Test basic TCP connectivity to a host:port"""
    try:
        logger.info(f"🔌 Testing TCP connection to {host}:{port}")

        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        start_time = time.time()
        result = sock.connect_ex((host, port))
        connect_time = time.time() - start_time

        sock.close()

        if result == 0:
            logger.info(".2f"            return True
        else:
            logger.error(f"❌ TCP connection failed to {host}:{port} (error code: {result})")
            return False

    except Exception as e:
        logger.error(f"❌ TCP connection error to {host}:{port}: {e}")
        return False

def test_multiple_ports(host, ports):
    """Test connectivity to multiple ports on the same host"""
    results = {}

    for port in ports:
        logger.info(f"\n🔍 Testing port {port} on {host}")
        result = asyncio.run(test_tcp_connectivity(host, port))
        results[port] = result

    return results

async def test_http_connectivity(host, port, timeout=10):
    """Test HTTP connectivity (basic GET request)"""
    try:
        import aiohttp

        url = f"http://{host}:{port}/"
        logger.info(f"🌐 Testing HTTP GET to {url}")

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            start_time = time.time()
            async with session.get(url) as response:
                response_time = time.time() - start_time

                logger.info(".2f"
                logger.info(f"   Status: {response.status}")
                logger.info(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")

                # Read a small amount of content
                content = await response.text()
                content_preview = content[:200] + "..." if len(content) > 200 else content
                logger.info(f"   Content Preview: {content_preview}")

                return response.status == 200

    except ImportError:
        logger.warning("⚠️ aiohttp not available, skipping HTTP test")
        return None
    except Exception as e:
        logger.error(f"❌ HTTP connection error to {host}:{port}: {e}")
        return False

def analyze_uid_142_issue():
    """Analyze the specific UID 142 connectivity issue"""

    logger.info("🔍 ANALYZING UID 142 CONNECTIVITY ISSUE")
    logger.info("="*60)

    # From the logs, UID 142 had these errors:
    # - Service unavailable at 129.146.3.173:8092/PullTask
    # - Request timeout after 40.0 seconds

    host = "129.146.3.173"
    ports_to_test = [8092, 80, 443]  # Bittensor port + common web ports

    logger.info(f"📊 Testing connectivity to UID 142's IP: {host}")
    logger.info("   From logs: Service unavailable at 129.146.3.173:8092/PullTask"    logger.info("   From logs: Request timeout after 40.0 seconds"
    # Test TCP connectivity
    tcp_results = test_multiple_ports(host, ports_to_test)

    # Test HTTP connectivity on port 8092
    if tcp_results.get(8092):
        logger.info("
🌐 Testing HTTP connectivity on port 8092..."        http_result = asyncio.run(test_http_connectivity(host, 8092))
        if http_result is False:
            logger.error("❌ HTTP test failed - this suggests the Bittensor validator service is not responding")
        elif http_result is None:
            logger.info("⚠️ HTTP test skipped (aiohttp not available)")

    # Analysis
    logger.info("
🎯 ANALYSIS:"    logger.info("-" * 40)

    if not tcp_results.get(8092):
        logger.error("❌ PORT 8092 BLOCKED: TCP connection to port 8092 failed")
        logger.error("   This means either:")
        logger.error("   1. Firewall is blocking port 8092")
        logger.error("   2. The validator service is not running")
        logger.error("   3. The IP address has changed")

    if tcp_results.get(80) or tcp_results.get(443):
        logger.info("✅ OTHER PORTS WORK: Ports 80/443 are accessible")
        logger.info("   This suggests the server is online but Bittensor service is down")

    logger.info("
💡 RECOMMENDATIONS:"    logger.info("-" * 40)
    logger.info("1. Check if UID 142's IP/port has changed")
    logger.info("2. Contact the validator operator about the service outage")
    logger.info("3. Try different validators while this one is fixed")
    logger.info("4. The subnet owner suggestion about different machine may be correct")

if __name__ == "__main__":
    analyze_uid_142_issue()

