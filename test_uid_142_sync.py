#!/usr/bin/env python3
"""
Synchronous network connectivity test for UID 142
Tests basic TCP connectivity without async complications
"""

import socket
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tcp_connectivity(host, port, timeout=10):
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
        result = test_tcp_connectivity(host, port)
        results[port] = result
        time.sleep(1)  # Small delay between tests

    return results

def analyze_uid_142_issue():
    """Analyze the specific UID 142 connectivity issue"""

    logger.info("🔍 ANALYZING UID 142 CONNECTIVITY ISSUE")
    logger.info("="*60)

    # From the logs, UID 142 had these errors:
    # - Service unavailable at 129.146.3.173:8092/PullTask
    # - Request timeout after 40.0 seconds

    host = "129.146.3.173"
    ports_to_test = [8092, 80, 443, 22]  # Bittensor port + common ports

    logger.info(f"📊 Testing connectivity to UID 142's IP: {host}")
    logger.info("   From logs: Service unavailable at 129.146.3.173:8092/PullTask"    logger.info("   From logs: Request timeout after 40.0 seconds"
    # Test TCP connectivity
    tcp_results = test_multiple_ports(host, ports_to_test)

    # Analysis
    logger.info("
🎯 ANALYSIS:"    logger.info("-" * 40)

    if not tcp_results.get(8092):
        logger.error("❌ PORT 8092 BLOCKED: TCP connection to port 8092 failed")
        logger.error("   This means either:")
        logger.error("   1. Firewall is blocking port 8092")
        logger.error("   2. The validator service is not running")
        logger.error("   3. The IP address has changed")
        logger.error("   4. Network routing issues")

    if tcp_results.get(80) or tcp_results.get(443):
        logger.info("✅ WEB PORTS WORK: Ports 80/443 are accessible")
        logger.info("   This suggests the server is online but Bittensor service is down")

    if tcp_results.get(22):
        logger.info("✅ SSH WORKS: Port 22 is accessible")
        logger.info("   Server is definitely online and reachable")

    # Specific recommendations
    logger.info("
💡 RECOMMENDATIONS:"    logger.info("-" * 40)

    if not tcp_results.get(8092):
        logger.info("1. 🔴 CRITICAL: Bittensor port 8092 is not accessible")
        logger.info("2. 📞 Contact UID 142 validator operator about service status")
        logger.info("3. 🔄 Use other validators while this one is being fixed")
        logger.info("4. 🖥️  Consider subnet owner suggestion about different machine")
        logger.info("5. 🌐 Check if IP address has changed (validator moved)")

    logger.info("
📋 NEXT STEPS:"    logger.info("-" * 40)
    logger.info("1. Test with different network/machine as suggested")
    logger.info("2. Monitor UID 142 status over next few hours")
    logger.info("3. Consider temporary validator blacklisting if persistent")
    logger.info("4. Report issue to Bittensor community if widespread")

if __name__ == "__main__":
    analyze_uid_142_issue()

