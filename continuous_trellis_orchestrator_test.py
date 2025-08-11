#!/usr/bin/env python3
"""
Continuous TRELLIS Orchestrator (Test)
- Targets Bittensor testnet (netuid 89)
- Uses separate log and DB files to avoid mixing with production
- Generates via http://localhost:8096

Run:
  python continuous_trellis_orchestrator_test.py --continuous (via run_trellis_mining_test.sh)
"""

import asyncio
import argparse
import logging
import sys
from typing import Any, Dict

# Import main orchestrator and components
from continuous_trellis_orchestrator import (
    ContinuousTrellisOrchestrator,
    TaskDatabase,
    logger as base_logger,
)

# Reconfigure logging to use test-specific file
for h in list(base_logger.handlers):
    base_logger.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_trellis_test.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class TaskDatabaseTest(TaskDatabase):
    def __init__(self, db_path: str = "continuous_trellis_tasks_test.db"):
        super().__init__(db_path=db_path)


class ContinuousTrellisOrchestratorTest(ContinuousTrellisOrchestrator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Override the database with a test-specific DB
        self.db = TaskDatabaseTest()
        self.logger.info("🧪 Using test database: continuous_trellis_tasks_test.db")

    def _get_default_config(self) -> Dict[str, Any]:
        cfg = super()._get_default_config()
        # Testnet defaults
        cfg.update({
            'wallet_name': 'manbeast3b',
            'hotkey_name': 'm3b',
            'netuid': 89,  # testnet mirror of SN17
            'min_validator_stake': 1.0,
            'generation_server_url': 'http://localhost:8096',
            'validation_server_url': 'http://localhost:10006',
            'output_dir': './continuous_trellis_outputs_test',
        })
        return cfg


async def main():
    parser = argparse.ArgumentParser(description="Continuous TRELLIS Orchestrator (Test)")
    parser.add_argument("--no-harvest", action="store_true", help="Disable task harvesting")
    parser.add_argument("--no-validate", action="store_true", help="Disable validation")
    parser.add_argument("--no-submit", action="store_true", help="Disable result submission")
    parser.add_argument("--generation-server", default="http://localhost:8096", help="TRELLIS generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    parser.add_argument("--output-dir", default="./continuous_trellis_outputs_test", help="Output directory (test)")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum local validation score")

    # Determinism
    parser.add_argument("--variable-seeds", action="store_true", help="Use prompt-hash based seeds (default: fixed 42)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed when not using variable seeds")

    args = parser.parse_args()

    config: Dict[str, Any] = {}
    if args.no_harvest:
        config['harvest_tasks'] = False
    if args.no_validate:
        config['validate_generations'] = False
    if args.no_submit:
        config['submit_results'] = False

    config['generation_server_url'] = args.generation_server
    config['validation_server_url'] = args.validation_server
    config['output_dir'] = args.output_dir
    config['min_local_score'] = args.min_score
    if args.variable_seeds:
        config['use_fixed_seed'] = False
    config['fixed_seed_value'] = args.seed

    orch = ContinuousTrellisOrchestratorTest(config)

    try:
        await orch.continuous_mining_loop()
    except Exception as e:
        logger.error(f"❌ Test orchestrator failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 