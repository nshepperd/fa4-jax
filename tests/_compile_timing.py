"""Opt-in pytest plugin: where does the suite's compile time go?

    uv run pytest tests --compile-timing ...

Times every cutejax kernel build (trace + lower + link, i.e. each call of
cutejax.compiler.compile_tvm_ffi) and collects jax.monitoring duration
events (XLA backend compile -- which includes GPU autotuning -- plus JAX's
own tracing and lowering), then prints a breakdown at session end next to
the total wall time. The XLA lowering duration includes any cutejax builds
triggered from inside a lowering rule, so the cutejax row is reported on its
own and not subtracted from anything.
"""

from __future__ import annotations

import collections
import functools
import time

_t0 = time.perf_counter()
_events: dict[str, list[float]] = collections.defaultdict(list)


def pytest_configure(config):
    import jax.monitoring

    def on_duration(event: str, duration: float, **kwargs):
        _events[event].append(duration)

    jax.monitoring.register_event_duration_secs_listener(on_duration)

    from cutejax import compiler

    orig = compiler.compile_tvm_ffi

    @functools.wraps(orig)
    def timed(*args, **kwargs):
        t = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            _events["cutejax/compile_tvm_ffi"].append(time.perf_counter() - t)

    compiler.compile_tvm_ffi = timed


def pytest_terminal_summary(terminalreporter):
    wall = time.perf_counter() - _t0
    tr = terminalreporter
    tr.section("compile timing")
    tr.write_line(f"session wall time: {wall / 60:.1f} min")
    tr.write_line(f"{'event':62} {'n':>6} {'total s':>9} {'% wall':>7} {'mean s':>8}")
    for name, ds in sorted(_events.items(), key=lambda kv: -sum(kv[1])):
        total = sum(ds)
        if total < 1:
            continue
        tr.write_line(f"{name[-62:]:62} {len(ds):6d} {total:9.1f} {100 * total / wall:6.1f}% {total / len(ds):8.3f}")
