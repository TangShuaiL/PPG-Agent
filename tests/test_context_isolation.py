"""Tests for request-scoped PPGContext isolation (contextvars).

Verifies that concurrent asyncio tasks / threads never share signal state,
which is the prerequisite for serving multiple web requests in one process.
"""
import asyncio
import concurrent.futures
import contextvars
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ppg_agent.context import (
    PPGContext, fresh_async_task_context, get_context, reset_context, set_context,
    spawn_isolated_task,
)


def test_lazy_create_and_reset():
    ctx1 = get_context()
    ctx1.age = 55.0
    assert get_context() is ctx1  # same scope → same object
    ctx2 = reset_context()
    assert ctx2 is not ctx1
    assert ctx2.age is None
    assert get_context() is ctx2


def test_async_task_isolation():
    """Concurrent asyncio tasks must not see each other's context.

    The web-layer pattern has TWO parts (both required):
      1. create_task(..., context=fresh context) — isolate the var binding;
      2. reset_context() at the start of each request — install a FRESH
         PPGContext object (copy_context alone only copies bindings, so a
         pre-existing context object would still be shared).

    Uses the production helper spawn_isolated_task（3.10/3.11+ 双兼容，
    torch_env 运行时 3.10 无 create_task 的 context= 形参）。
    """
    # Simulate leftover state from a previous request in the outer scope
    get_context().age = 999.0
    results: dict[str, object] = {}

    async def worker(name: str, age: float, delay: float):
        ctx = get_context()  # already fresh: installed by fresh_async_task_context()
        ctx.age = age
        ctx.ppg_signal = np.ones(100) * age
        await asyncio.sleep(delay)  # interleave with the other task
        ctx = get_context()  # re-fetch after yielding control
        # Assertions must run INSIDE the task: after asyncio.run() ends,
        # get_context() resolves in the outer scope again.
        results[name] = ctx
        assert ctx.age == age, (name, ctx.age)
        assert ctx.ppg_signal.mean() == age
        assert ctx is get_context()

    async def main():
        # Web-layer pattern: one fresh context per request/task
        t1 = spawn_isolated_task(worker("A", 30.0, 0.02))
        t2 = spawn_isolated_task(worker("B", 70.0, 0.01))
        await asyncio.gather(t1, t2)

    asyncio.run(main())
    assert len(results) == 2
    assert results["A"] is not results["B"]  # distinct context objects
    assert get_context().age == 999.0  # outer scope untouched


def test_thread_isolation_with_explicit_context():
    """Web layer installs an explicit context; worker threads get their own."""
    main_ctx = PPGContext()
    main_ctx.age = 99.0
    set_context(main_ctx)

    def thread_fn():
        # Fresh thread → contextvars default → lazy-created fresh context
        ctx = get_context()
        return ctx.age  # must NOT see main thread's 99.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        ages = list(ex.map(lambda _: thread_fn(), range(4)))

    assert all(a is None for a in ages), ages
    assert get_context().age == 99.0  # main thread unaffected


def test_context_propagation_via_copy_context():
    """copy_context().run() propagates state into a worker thread (documented path)."""
    src = PPGContext()
    src.age = 42.0
    set_context(src)

    def read_age():
        return get_context().age

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        copied = contextvars.copy_context()
        fut = ex.submit(copied.run, read_age)
        assert fut.result() == 42.0


if __name__ == "__main__":
    test_lazy_create_and_reset()
    test_async_task_isolation()
    test_thread_isolation_with_explicit_context()
    test_context_propagation_via_copy_context()
    print("All context isolation tests passed.")
