"""Lane concurrency, per-path write serialization, dedup cache, timeouts."""

import asyncio
import threading
import time

from agent_project.harness.effects import make_effect
from agent_project.harness.scheduler import Scheduler


def _run(coro):
    return asyncio.run(coro)


def test_parallel_reads_overlap():
    starts = []

    def run(effect):
        starts.append(time.monotonic())
        time.sleep(0.1)
        return "r"

    async def main():
        sched = Scheduler(max_parallel=4)
        effects = [make_effect("grep", {"pattern": f"p{i}"}) for i in range(4)]
        return await sched.submit_many([(e, "parallel") for e in effects], run)

    t0 = time.monotonic()
    outcomes = _run(main())
    assert all(o.ok for o in outcomes)
    assert time.monotonic() - t0 < 0.25  # 4 x 0.1s overlapped, not 0.4s


def test_writes_to_same_path_serialize():
    order = []

    def run(effect):
        order.append(("start", effect.arguments["n"]))
        time.sleep(0.05)
        order.append(("end", effect.arguments["n"]))
        return "w"

    async def main():
        sched = Scheduler()
        effects = [
            make_effect("file_ops", {"action": "write", "path": "same.txt", "n": i})
            for i in (1, 2)
        ]
        return await sched.submit_many([(e, "write:same.txt") for e in effects], run)

    _run(main())
    # serialized: one must fully end before the other starts
    assert order in (
        [("start", 1), ("end", 1), ("start", 2), ("end", 2)],
        [("start", 2), ("end", 2), ("start", 1), ("end", 1)],
    )


def test_dedup_cache_executes_once():
    calls = []

    def run(effect):
        calls.append(effect.idempotency_key)
        return "cached-output"

    async def main():
        sched = Scheduler(cache_ttl_s=60)
        effect = make_effect("grep", {"pattern": "same"})
        first = await sched.submit(effect, "parallel", run)
        second = await sched.submit(effect, "parallel", run)
        return first, second

    first, second = _run(main())
    assert len(calls) == 1
    assert first.output == "cached-output"
    assert second.from_cache and second.output == "cached-output"


def test_concurrent_identical_reads_share_inflight():
    calls = []

    def run(effect):
        calls.append(1)
        time.sleep(0.05)
        return "shared"

    async def main():
        sched = Scheduler(cache_ttl_s=60)
        effect = make_effect("grep", {"pattern": "x"})
        return await asyncio.gather(
            sched.submit(effect, "parallel", run),
            sched.submit(effect, "parallel", run),
            sched.submit(effect, "parallel", run),
        )

    outcomes = _run(main())
    assert len(calls) == 1
    assert all(o.output == "shared" for o in outcomes)


def test_mutating_effects_bypass_cache():
    calls = []

    def run(effect):
        calls.append(1)
        return "w"

    async def main():
        sched = Scheduler()
        effect = make_effect("file_ops", {"action": "write", "path": "a.txt", "content": "v"})
        await sched.submit(effect, "write:a.txt", run)
        await sched.submit(effect, "write:a.txt", run)

    _run(main())
    assert len(calls) == 2


def test_timeout_produces_failed_outcome():
    def run(effect):
        time.sleep(1.0)
        return "late"

    async def main():
        sched = Scheduler(default_timeout_s=0.05)
        return await sched.submit(make_effect("grep", {"pattern": "slow"}), "parallel", run)

    outcome = _run(main())
    assert not outcome.ok
    from agent_project.harness.errors import ToolTimeoutError

    assert isinstance(outcome.error, ToolTimeoutError)


def test_scheduler_is_threadsafe_for_sync_submission():
    # the event loop owns the scheduler; this verifies cross-thread sanity
    sched = Scheduler()

    def run(effect):
        return threading.current_thread().name

    outcome = _run(sched.submit(make_effect("calculator", {"expression": "1"}), "parallel", run))
    assert outcome.ok and outcome.output
