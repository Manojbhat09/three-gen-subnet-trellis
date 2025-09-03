#!/usr/bin/env python3
"""
Network diagnostic script to identify connectivity issues
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and return result"""
    logger.info(f"🔍 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"✅ SUCCESS: {result.stdout.strip()}")
            return True, result.stdout.strip()
        else:
            logger.warning(f"❌ FAILED: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        return False, str(e)

def network_diagnostics():
    """Run comprehensive network diagnostics"""
    logger.info("🔍 NETWORK DIAGNOSTICS FOR UID 142 CONNECTIVITY")
    logger.info("="*60)

    target = "129.146.3.173"

    # Test 1: Basic connectivity
    run_command(f"ping -c 3 -W 2 {target}", "Testing basic ICMP connectivity")

    # Test 2: TCP connectivity
    run_command(f"nc -vz -w 5 {target} 8092", "Testing TCP port 8092 connectivity")

    # Test 3: Traceroute
    run_command(f"traceroute -m 10 -w 2 {target}", "Testing network path")

    # Test 4: DNS resolution
    run_command(f"nslookup {target}", "Testing DNS resolution")

    # Test 5: Check local network
    run_command("ip route show", "Checking local routing table")
    run_command("ip addr show", "Checking network interfaces")

    # Test 6: Check for MTU issues
    run_command(f"ping -c 3 -M do -s 1472 {target}", "Testing MTU/path MTU discovery")

def main():
    """Main diagnostic function"""
    logger.info("🌐 NETWORK DIAGNOSTIC TOOL")
    logger.info("This will help identify why HTTP requests to UID 142 timeout")

    network_diagnostics()

    logger.info("\n💡 POSSIBLE SOLUTIONS:")
    logger.info("-" * 40)
    logger.info("1. 🔄 Switch to working validators (81, 128, etc.)")
    logger.info("2. 🌐 Use VPN or different network interface")
    logger.info("3. ⏰ Increase timeout values in orchestrator")
    logger.info("4. 📞 Contact network administrator about routing")
    logger.info("5. 🔀 Use proxy or alternative connection method")

if __name__ == "__main__":
    main()

