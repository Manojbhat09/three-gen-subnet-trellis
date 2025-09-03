#!/usr/bin/env python3
"""
Comprehensive debug script for UID 142 connectivity issues
Tests multiple approaches and provides detailed analysis
"""

import socket
import time
import json
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tcp_connectivity(host, port, timeout=10):
    """Test basic TCP connectivity"""
    try:
        logger.info(f"🔌 Testing TCP to {host}:{port}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        start_time = time.time()
        result = sock.connect_ex((host, port))
        connect_time = time.time() - start_time

        sock.close()

        if result == 0:
            logger.info(".2f"            return {"success": True, "time": connect_time}
        else:
            logger.error(f"❌ TCP failed (code: {result})")
            return {"success": False, "error": f"Connection refused (code: {result})"}

    except socket.timeout:
        logger.error(f"❌ TCP timeout after {timeout}s")
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        logger.error(f"❌ TCP error: {e}")
        return {"success": False, "error": str(e)}

def test_network_diagnostics(host):
    """Run basic network diagnostics"""
    logger.info(f"🔍 Running network diagnostics for {host}")

    diagnostics = {}

    # Test ping
    try:
        result = subprocess.run(['ping', '-c', '3', '-W', '5', host],
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            diagnostics['ping'] = {"success": True, "output": result.stdout}
            logger.info("✅ Ping successful")
        else:
            diagnostics['ping'] = {"success": False, "error": result.stderr}
            logger.error("❌ Ping failed")
    except Exception as e:
        diagnostics['ping'] = {"success": False, "error": str(e)}
        logger.error(f"❌ Ping error: {e}")

    # Test traceroute (if available)
    try:
        result = subprocess.run(['traceroute', '-m', '10', '-w', '2', host],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            diagnostics['traceroute'] = {"success": True, "output": result.stdout}
            logger.info("✅ Traceroute completed")
        else:
            diagnostics['traceroute'] = {"success": False, "error": result.stderr}
    except Exception as e:
        diagnostics['traceroute'] = {"success": False, "error": str(e)}
        logger.warning(f"⚠️ Traceroute not available or failed: {e}")

    return diagnostics

def analyze_log_patterns():
    """Analyze patterns from the provided logs"""
    logger.info("📊 ANALYZING LOG PATTERNS FOR UID 142")

    # From the logs provided
    uid_142_errors = [
        "Service unavailable at 129.146.3.173:8092/PullTask",
        "Request timeout after 40.0 seconds",
        "TCP connection timeout",
        "HTTP 408 Request Timeout"
    ]

    analysis = {
        "ip_address": "129.146.3.173",
        "port": 8092,
        "common_errors": uid_142_errors,
        "error_patterns": [
            "Timeout errors (40s timeout)",
            "Service unavailable messages",
            "TCP connection issues"
        ]
    }

    logger.info(f"📍 Target: {analysis['ip_address']}:{analysis['port']}")
    logger.info("🚨 Error Patterns:")
    for error in analysis['error_patterns']:
        logger.info(f"   • {error}")

    return analysis

def comprehensive_uid_142_test():
    """Run comprehensive tests for UID 142"""

    logger.info("🔍 COMPREHENSIVE UID 142 CONNECTIVITY TEST")
    logger.info("="*60)

    # Analyze log patterns first
    log_analysis = analyze_log_patterns()

    host = log_analysis['ip_address']
    bittensor_port = log_analysis['port']

    # Test different ports
    test_ports = [bittensor_port, 80, 443, 22, 8080]

    logger.info("
🔌 PHASE 1: TCP Connectivity Tests"    logger.info("-" * 40)

    tcp_results = {}
    for port in test_ports:
        logger.info(f"\n📡 Testing port {port}...")
        result = test_tcp_connectivity(host, port, timeout=15)
        tcp_results[port] = result

    # Network diagnostics
    logger.info("
🌐 PHASE 2: Network Diagnostics"    logger.info("-" * 40)

    network_diag = test_network_diagnostics(host)

    # Analysis and recommendations
    logger.info("
🎯 PHASE 3: ANALYSIS & RECOMMENDATIONS"    logger.info("-" * 40)

    # Analyze Bittensor port specifically
    bittensor_result = tcp_results.get(bittensor_port, {})
    if not bittensor_result.get('success', False):
        logger.error("❌ CRITICAL: Bittensor port 8092 is NOT accessible")
        logger.error("   This explains the 'Service unavailable' errors")

        # Check if other ports work
        other_ports_working = any(
            tcp_results.get(port, {}).get('success', False)
            for port in test_ports if port != bittensor_port
        )

        if other_ports_working:
            logger.info("✅ OTHER PORTS: Some ports are accessible")
            logger.info("   Server is online, but Bittensor service is down")
        else:
            logger.error("❌ ALL PORTS: No ports are accessible")
            logger.error("   Server may be down or network blocked")

    # Ping analysis
    ping_result = network_diag.get('ping', {})
    if ping_result.get('success'):
        logger.info("✅ NETWORK: Server is reachable via ping")
    else:
        logger.error("❌ NETWORK: Server is NOT reachable via ping")
        logger.error("   This could be a routing or firewall issue")

    # Recommendations
    logger.info("
💡 RECOMMENDATIONS:"    logger.info("-" * 40)

    recommendations = [
        "1. 🔴 IMMEDIATE: Stop trying UID 142 until fixed",
        "2. 📞 CONTACT: Reach out to UID 142 validator operator",
        "3. 🔄 ALTERNATIVE: Use other working validators",
        "4. 🖥️  TEST: Try from different machine/network as suggested",
        "5. 📊 MONITOR: Check if IP address changed",
        "6. 🛡️  FALLBACK: Consider temporary blacklisting",
        "7. 📈 REPORT: Report to Bittensor community if widespread"
    ]

    for rec in recommendations:
        logger.info(rec)

    # Summary
    logger.info("
📋 SUMMARY:"    logger.info("-" * 40)
    logger.info(f"Target: {host}:{bittensor_port}")
    logger.info(f"Bittensor Port Working: {tcp_results.get(bittensor_port, {}).get('success', False)}")
    logger.info(f"Server Pingable: {network_diag.get('ping', {}).get('success', False)}")
    logger.info("Issue: Bittensor validator service is not responding"
    logger.info("Action: Use alternative validators while this is resolved"
    return {
        "tcp_results": tcp_results,
        "network_diagnostics": network_diag,
        "log_analysis": log_analysis,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    try:
        results = comprehensive_uid_142_test()

        # Save results
        with open('uid_142_debug_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("
📄 Results saved to uid_142_debug_results.json"    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


