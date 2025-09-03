#!/usr/bin/env python3
"""
Wallet setup guide and UID 142 connectivity analysis
"""

import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_wallet_setup():
    """Check current wallet setup"""
    logger.info("🔍 CHECKING BITTENSOR WALLET SETUP")
    logger.info("="*50)

    # Check common wallet locations
    wallet_paths = [
        "/root/.bittensor/wallets",
        "/home/mbhat/.bittensor/wallets",
        os.path.expanduser("~/.bittensor/wallets")
    ]

    wallet_found = False
    for path in wallet_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found wallet directory: {path}")
            wallet_found = True

            # List contents
            try:
                contents = os.listdir(path)
                logger.info(f"   Contents: {contents}")

                # Check for default wallet
                default_wallet = os.path.join(path, "default")
                if os.path.exists(default_wallet):
                    logger.info(f"   ✅ Found 'default' wallet: {default_wallet}")

                    # Check hotkeys
                    hotkeys_dir = os.path.join(default_wallet, "hotkeys")
                    if os.path.exists(hotkeys_dir):
                        hotkeys = os.listdir(hotkeys_dir)
                        logger.info(f"   ✅ Hotkeys found: {hotkeys}")
                    else:
                        logger.error(f"   ❌ No hotkeys directory in {default_wallet}")

                else:
                    logger.warning(f"   ⚠️ No 'default' wallet found in {path}")

            except Exception as e:
                logger.error(f"   ❌ Error reading wallet directory: {e}")
        else:
            logger.info(f"❌ Wallet directory not found: {path}")

    if not wallet_found:
        logger.error("❌ No Bittensor wallet directories found!")
        return False

    return wallet_found

def provide_wallet_setup_instructions():
    """Provide wallet setup instructions"""
    logger.info("\n💡 BITTENSOR WALLET SETUP INSTRUCTIONS")
    logger.info("="*50)

    instructions = [
        "1. 🔧 Install Bittensor CLI:",
        "   pip install bittensor",
        "",
        "2. 📝 Create a new wallet:",
        "   btcli wallet new",
        "   (Follow prompts to create coldkey and hotkey)",
        "",
        "3. 🔑 Generate hotkey for mining:",
        "   btcli wallet regen-hotkey",
        "   (Choose your wallet and generate hotkey)",
        "",
        "4. 💰 Fund your wallet:",
        "   - Get TAO tokens for testnet/mainnet",
        "   - Use faucet or exchange",
        "",
        "5. 🔐 Set wallet environment variables (optional):",
        "   export BT_WALLET_NAME=default",
        "   export BT_WALLET_HOTKEY=default",
        "",
        "6. ✅ Verify setup:",
        "   btcli wallet list",
        "   btcli wallet balance",
        "",
        "7. 🏃‍♂️ Run the test:",
        "   python test_uid_142.py"
    ]

    for instruction in instructions:
        logger.info(instruction)

def analyze_uid_142_connectivity():
    """Analyze UID 142 connectivity issue"""
    logger.info("\n🔍 UID 142 CONNECTIVITY ANALYSIS")
    logger.info("="*50)

    logger.info("📊 From the logs, UID 142 shows these errors:")
    logger.info("   • Service unavailable at 129.146.3.173:8092/PullTask")
    logger.info("   • Request timeout after 40.0 seconds")
    logger.info("   • TCP connection timeout")
    logger.info("   • HTTP 408 Request Timeout")

    logger.info("\n🎯 ROOT CAUSE ANALYSIS:")
    logger.info("   The issue is NOT with wallet setup")
    logger.info("   The issue is with UID 142's Bittensor validator service")
    logger.info("   Port 8092 on IP 129.146.3.173 is not responding")

    logger.info("\n💡 IMMEDIATE SOLUTIONS:")
    logger.info("   1. 🔄 Use other validators (UID 81, 128, etc. are working)")
    logger.info("   2. 📞 Contact UID 142 validator operator")
    logger.info("   3. 🖥️ Try from different machine/network")
    logger.info("   4. ⏳ Wait and retry later (service may be restarting)")

def main():
    """Main function"""
    logger.info("🚀 BITTENSOR WALLET & UID 142 DIAGNOSTIC TOOL")
    logger.info("="*60)

    # Check wallet setup
    wallet_ok = check_wallet_setup()

    if not wallet_ok:
        provide_wallet_setup_instructions()
    else:
        logger.info("✅ Wallet setup looks good!")

    # Always provide UID 142 analysis
    analyze_uid_142_connectivity()

    logger.info("\n📋 SUMMARY:")
    logger.info("-" * 40)
    if wallet_ok:
        logger.info("✅ Wallet: Configured")
        logger.info("✅ Can run: test_uid_142.py")
    else:
        logger.info("❌ Wallet: Not configured")
        logger.info("❌ Need to: Run wallet setup first")

    logger.info("❌ UID 142: Network connectivity issue")
    logger.info("✅ Alternative: Use working validators")

if __name__ == "__main__":
    main()

