# quick_dual_validation_test.py
import asyncio
from types import SimpleNamespace

# Import the module under test
import continuous_trellis_orchestrator_lora_test as mod

# 1) Monkeypatch remote validator call to return a fake score
class FakeResp:
    status_code = 200
    def json(self):
        return {"score": 0.81234}
def fake_post(url, json=None, timeout=10):
    return FakeResp()
mod.requests.post = fake_post  # patch requests.post used in the module

# 2) Monkeypatch local production-accurate validator
def fake_local_validate(spz_bytes: bytes, prompt: str):
    return {
        "validation_engine_score": 0.79567,
        "alignment_score": 0.6123,
        "quality_score": 0.9234,
    }
mod.local_prod_validate = fake_local_validate

# 3) Monkeypatch base submit_result to avoid network and return quickly
Base = mod.ContinuousTrellisOrchestrator
async def fake_base_submit_result(self, task, generation_result, retry=False):
    # simulate fast submit that succeeds
    task.submitted_at = 123.0
    task.submission_success = True
    return True
Base.submit_result = fake_base_submit_result

async def main():
    # 4) Build orchestrator with dual validation enabled
    orch = mod.ContinuousTrellisOrchestratorLoRATest({
        "dual_validation": True,
        "submit_results": True,
        "validate_generations": False,  # we test post-submit dual path
        "generation_server_url": "http://localhost:8096",
        "validation_server_url": "http://localhost:10006",
    })

    # 5) Fake task and generation result
    task = SimpleNamespace(
        task_id="t1",
        prompt="silver robot wearing green scarf",
        generation_time=12.34,
        validation_time=1.23,
        local_validation_score=None,
    )
    generation_result = {"ply_data": b"FAKE_SPZ_BYTES"}

    # 6) Run submit_result (will perform remote+local validation and print a table)
    await orch.submit_result(task, generation_result)

if __name__ == "__main__":
    asyncio.run(main())
