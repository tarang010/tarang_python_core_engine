"""
Tarang 2.3.0 — worker_pool.py
===============================
Process pool for CPU-intensive tasks (TTS, modulation, extraction, MCQ).
Keeps the FastAPI event loop free — heavy work runs in separate OS processes.

v2.3.0 optimizations vs v2.2.0:
  FIX A — RETAINED: asyncio.wait_for(sem.acquire(), timeout=0) always raises
           TimeoutError even when the semaphore has free slots. Uses synchronous
           peek on sem._value instead — canonical asyncio pattern, atomic in
           single-threaded event loop.

  FIX B — _queue_depth counter managed correctly inside semaphore boundary.
           depth == (MAX_QUEUE - sem._value) invariant always holds.

  FIX C — get_pool_stats() does NOT call _get_pool(), avoiding lazy pool
           spawn on bare /status health-checks.

  OPT 1 — _noop_warmup() added: a trivial function that triggers worker
           process startup + heavy module pre-imports. Called by bridge.py
           startup_event via _warm_up_pool() to eliminate cold-start latency
           on the first real request.

  OPT 2 — _worker_init() now pre-imports file5_mcq and file6_analytics in
           addition to the original four modules, so all heavy imports are
           done at pool startup, not first use.

  OPT 3 — run_in_process() accepts an optional bypass_pool=True flag that
           runs the function in the default ThreadPoolExecutor instead of
           the process pool. Used by bridge.py for I/O-light tasks like
           captions that don't need a separate process.

Usage:
    from worker_pool import run_in_process, get_pool_stats, shutdown_pool

    # CPU-heavy (default):
    result = await run_in_process(some_sync_function, arg1, kwarg=val)

    # I/O-light bypass (skips process pool, uses thread pool):
    result = await run_in_process(lightweight_fn, arg1, bypass_pool=True)
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

logger = logging.getLogger("worker_pool")

# ── Configuration ─────────────────────────────────────────────────────────────
CPU_WORKERS: int = int(os.getenv("TARANG_CPU_WORKERS", str(max(2, (os.cpu_count() or 4)))))
MAX_QUEUE:   int = int(os.getenv("TARANG_MAX_QUEUE",   "50"))

logger.info(f"[worker_pool] CPU_WORKERS={CPU_WORKERS} | MAX_QUEUE={MAX_QUEUE}")

# ── Globals ────────────────────────────────────────────────────────────────────
_pool:        Optional[ProcessPoolExecutor] = None
_thread_pool: Optional[ThreadPoolExecutor]  = None   # OPT 3: for bypass_pool tasks
_sem:         Optional[asyncio.Semaphore]   = None
_queue_depth: int = 0   # guarded by the event loop (single-threaded async)


def _get_pool() -> ProcessPoolExecutor:
    """Return the singleton process pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=CPU_WORKERS,
            initializer=_worker_init,
        )
        logger.info(f"[worker_pool] ProcessPoolExecutor started | workers={CPU_WORKERS}")
    return _pool


def _get_thread_pool() -> ThreadPoolExecutor:
    """Return the singleton thread pool for I/O-light bypass tasks."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 4) * 4),
            thread_name_prefix="tarang_io",
        )
        logger.info("[worker_pool] ThreadPoolExecutor started for bypass tasks")
    return _thread_pool


def _worker_init():
    """
    Called once in each worker process on startup.
    Pre-imports ALL heavy modules so the first real task isn't slow.

    OPT 2: Now includes file5_mcq and file6_analytics which were missing
    from v2.2.0, causing the first MCQ or analytics call to pay full import
    cost inside a request.
    """
    import logging as _log
    _log.basicConfig(level=logging.INFO)
    _wlog = _log.getLogger("tarang_worker")
    _wlog.info(f"[worker] PID {os.getpid()} initialised")
    try:
        import file1_extractor   # noqa: F401
        import file2_tts         # noqa: F401
        import file3_modulator   # noqa: F401
        import file7_captions    # noqa: F401
        _wlog.info(f"[worker] PID {os.getpid()} core modules pre-loaded")
    except ImportError as e:
        _wlog.warning(f"[worker] Pre-import warning: {e}")
    try:
        import file5_mcq         # noqa: F401  OPT 2
        import file6_analytics   # noqa: F401  OPT 2
        _wlog.info(f"[worker] PID {os.getpid()} MCQ/analytics modules pre-loaded")
    except ImportError as e:
        _wlog.warning(f"[worker] MCQ/analytics pre-import warning: {e}")


def _get_semaphore() -> asyncio.Semaphore:
    """Return the singleton semaphore (must be called from an async context)."""
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_QUEUE)
    return _sem


# ── OPT 1: noop warmup task ───────────────────────────────────────────────────

def _noop_warmup() -> dict:
    """
    Trivial function submitted to the process pool at startup.
    Its only purpose is to force the worker process to start and run
    _worker_init(), pre-loading all heavy modules before any user request
    arrives. Returns immediately after import.

    Called by bridge.py _warm_up_pool() via run_in_process().
    """
    import os as _os
    return {"warmed": True, "pid": _os.getpid()}


# ── Public API ─────────────────────────────────────────────────────────────────

async def run_in_process(fn: Callable, *args, bypass_pool: bool = False, **kwargs) -> Any:
    """
    Run a synchronous function in the process pool (default) or thread pool
    (bypass_pool=True for I/O-light tasks).

    OPT 3: bypass_pool=True skips the process pool semaphore entirely and
    runs the function in the shared thread pool. Use for tasks like caption
    generation that are pure Python and don't benefit from a separate process.

    FIX A: Correct non-blocking semaphore pattern — synchronous peek on
    sem._value, then acquire only if a slot is confirmed free. asyncio is
    single-threaded so the peek+acquire sequence is atomic.

    Raises:
        RuntimeError("queue_full")  — when MAX_QUEUE slots are all occupied
        Any exception raised by fn  — propagated normally
    """
    global _queue_depth

    # OPT 3: bypass_pool path — direct thread pool, no semaphore overhead
    if bypass_pool:
        loop = asyncio.get_event_loop()
        pool = _get_thread_pool()
        t0 = time.perf_counter()
        try:
            if kwargs:
                bound  = partial(fn, **kwargs)
                result = await loop.run_in_executor(pool, bound, *args)
            else:
                result = await loop.run_in_executor(pool, fn, *args)
            logger.info(f"[worker_pool] {fn.__name__} (bypass) completed in {time.perf_counter()-t0:.2f}s")
            return result
        except Exception as exc:
            logger.error(f"[worker_pool] {fn.__name__} (bypass) FAILED: {type(exc).__name__}: {exc}")
            raise

    # Standard process pool path
    sem = _get_semaphore()

    # FIX A: synchronous peek — atomic because asyncio is single-threaded.
    if sem._value == 0:
        stats = get_pool_stats()
        logger.warning(
            f"[worker_pool] Queue full | depth={stats['queue_depth']} | "
            f"max={stats['max_queue']} | fn={fn.__name__}"
        )
        raise RuntimeError("queue_full")

    await sem.acquire()

    _queue_depth += 1
    loop = asyncio.get_event_loop()
    pool = _get_pool()
    t0   = time.perf_counter()

    try:
        if kwargs:
            bound  = partial(fn, **kwargs)
            result = await loop.run_in_executor(pool, bound, *args)
        else:
            result = await loop.run_in_executor(pool, fn, *args)

        elapsed = time.perf_counter() - t0
        logger.info(f"[worker_pool] {fn.__name__} completed in {elapsed:.2f}s")
        return result

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(
            f"[worker_pool] {fn.__name__} FAILED after {elapsed:.2f}s | "
            f"{type(exc).__name__}: {exc}"
        )
        raise

    finally:
        _queue_depth -= 1
        sem.release()


def get_pool_stats() -> dict:
    """
    Return current pool stats for the /status endpoint.
    FIX C: Does NOT call _get_pool() — avoids lazily spawning the pool on a
    bare health-check before any work has been submitted.
    """
    sem = _sem  # peek without creating
    return {
        "cpu_workers":   CPU_WORKERS,
        "max_queue":     MAX_QUEUE,
        "queue_depth":   _queue_depth,
        "slots_free":    (sem._value if sem else MAX_QUEUE),
        "pool_pid":      os.getpid(),
        "pool_started":  _pool is not None,
    }


async def shutdown_pool():
    """Graceful shutdown — call from FastAPI lifespan/shutdown event."""
    global _pool, _thread_pool
    if _pool:
        logger.info("[worker_pool] Shutting down ProcessPoolExecutor...")
        _pool.shutdown(wait=True)
        _pool = None
        logger.info("[worker_pool] Process pool shut down cleanly.")
    if _thread_pool:
        logger.info("[worker_pool] Shutting down ThreadPoolExecutor...")
        _thread_pool.shutdown(wait=False)
        _thread_pool = None