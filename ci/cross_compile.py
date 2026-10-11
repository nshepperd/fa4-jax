"""Compile (but never launch) the kernels for another GPU generation.

    uv run python ci/cross_compile.py 90    # Hopper wgmma kernels, from any GPU box
    uv run python ci/cross_compile.py 100   # datacenter-Blackwell tcgen05 kernels

The CuTe DSL kernels are traced and compiled while JAX traces/lowers a call,
so `jax.jit(f).lower(...)` builds every kernel a call would use without
running anything. With fa4-jax's device detection patched to report the
target and CUTE_DSL_ARCH set to match, that catches tracing/compile errors
on sm_90/sm_100 code paths -- e.g. an API change in a new cutlass-dsl --
on a machine without the hardware, before paying for ci/runpod_ci.py.
Numerics are not checked; that still needs the real GPU.
"""

from __future__ import annotations

import os
import sys
import traceback

ARCH = int(sys.argv[1]) if len(sys.argv) > 1 else 100
os.environ["CUTE_DSL_ARCH"] = f"sm_{ARCH}a"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fa4_jax.kernels as K  # noqa: E402
from fa4_jax import causal_mask, document_mask, flash_attn, flash_attn_varlen  # noqa: E402

K._device_arch = lambda: ARCH

# (batch, s_q, s_kv, heads, kv_heads, d, d_v) -- the test suite's shape matrix
# in miniature: head dims small/odd/large, mixed d/d_v, GQA.
SHAPES = [
    (1, 128, 128, 2, 2, 64, 64),
    (1, 128, 128, 2, 2, 128, 128),
    (1, 128, 128, 2, 2, 96, 96),
    (1, 128, 128, 2, 2, 16, 16),
    (1, 128, 128, 2, 2, 48, 48),
    (1, 128, 128, 2, 2, 64, 128),
    (1, 128, 128, 2, 2, 64, 32),
    (1, 128, 128, 4, 2, 64, 64),
    (2, 128, 128, 4, 2, 48, 16),
    (1, 96, 160, 2, 1, 192, 192),
    (1, 96, 160, 2, 1, 256, 256),
    (1, 96, 160, 2, 1, 256, 128),
    (1, 96, 160, 2, 1, 224, 224),
]


def sds(*shape, dtype=jnp.bfloat16):
    return jax.ShapeDtypeStruct(shape, dtype)


def cases():
    for b, sq, skv, h, hkv, d, dv in SHAPES:
        q, k, v = sds(b, sq, h, d), sds(b, skv, hkv, d), sds(b, skv, hkv, dv)
        tag = f"b{b} s{sq}/{skv} h{h}/{hkv} d{d}/{dv}"
        for causal in (False, True):
            kw = dict(backend="cute", causal=causal)
            yield f"fwd {tag} causal={causal}", lambda q, k, v, kw=kw: flash_attn(q, k, v, **kw), (q, k, v)
            yield f"lse {tag} causal={causal}", lambda q, k, v, kw=kw: flash_attn(q, k, v, return_lse=True, **kw), (q, k, v)
            yield (
                f"grad {tag} causal={causal}",
                lambda q, k, v, kw=kw: jax.grad(
                    lambda q, k, v: flash_attn(q, k, v, **kw).astype(jnp.float32).sum(), argnums=(0, 1, 2)
                )(q, k, v),
                (q, k, v),
            )
            yield (
                f"jvp {tag} causal={causal}",
                lambda q, k, v, kw=kw: jax.jvp(lambda q, k, v: flash_attn(q, k, v, **kw), (q, k, v), (q, k, v)),
                (q, k, v),
            )
    q, k, v = sds(1, 128, 2, 64), sds(1, 128, 2, 64), sds(1, 128, 2, 64)
    for name, kw in [
        ("window", dict(window_size=(32, 16))),
        ("mask_mod causal", dict(mask_mod=causal_mask())),
        ("mask_mod causal + window", dict(mask_mod=causal_mask(), window_size=(32, 0))),
    ]:
        kw = dict(backend="cute", **kw)
        yield f"fwd {name}", lambda q, k, v, kw=kw: flash_attn(q, k, v, **kw), (q, k, v)
        yield (
            f"grad {name}",
            lambda q, k, v, kw=kw: jax.grad(
                lambda q, k, v: flash_attn(q, k, v, **kw).astype(jnp.float32).sum(), argnums=(0, 1, 2)
            )(q, k, v),
            (q, k, v),
        )
    doc = sds(1, 128, dtype=jnp.int32)
    yield (
        "fwd document_mask",
        lambda q, k, v, doc: flash_attn(q, k, v, backend="cute", mask_mod=document_mask(doc)),
        (q, k, v, doc),
    )
    cu = sds(3, dtype=jnp.int32)
    for h, hkv, d in [(2, 2, 64), (4, 2, 64), (2, 1, 128)]:
        qp, kp, vp = sds(256, h, d), sds(256, hkv, d), sds(256, hkv, d)
        for causal in (False, True):
            kw = dict(backend="cute", causal=causal)
            tag = f"varlen h{h}/{hkv} d{d} causal={causal}"
            yield f"fwd {tag}", lambda q, k, v, cu, kw=kw: flash_attn_varlen(q, k, v, cu, cu, **kw), (qp, kp, vp, cu)
            yield (
                f"grad {tag}",
                lambda q, k, v, cu, kw=kw: jax.grad(
                    lambda q, k, v: flash_attn_varlen(q, k, v, cu, cu, **kw).astype(jnp.float32).sum(),
                    argnums=(0, 1, 2),
                )(q, k, v),
                (qp, kp, vp, cu),
            )
            yield (
                f"jvp {tag}",
                lambda q, k, v, cu, kw=kw: jax.jvp(
                    lambda q, k, v: flash_attn_varlen(q, k, v, cu, cu, **kw), (q, k, v), (q, k, v)
                ),
                (qp, kp, vp, cu),
            )


def main():
    failed = rejected = ok = 0
    for name, fn, args in cases():
        try:
            jax.jit(fn).lower(*args)
            ok += 1
        except ValueError as e:
            if "cannot run head_dim" in str(e):
                rejected += 1  # a clean, intended rejection (e.g. hd > 128 on sm_100)
                continue
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        except Exception as e:  # noqa: BLE001 -- report every compile failure
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
    print(f"sm_{ARCH}: {ok} compiled, {rejected} cleanly rejected, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
