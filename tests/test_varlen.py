"""Varlen (packed) attention: semantics, differentiation wiring, and kernels.

Anchoring strategy: the varlen *reference* is checked against a per-sequence
loop over the (independently tested) dense reference — that pins down the
packing semantics (segment isolation, per-sequence bottom-right alignment,
sequence-local mask_mod indices, zero-length sequences). The cute kernels are
then checked against the varlen reference for values, gradients, and the LSE.

Reference/wiring tests run anywhere; kernel tests are marked ``cuda``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_close

import fa4_jax
from fa4_jax import (
    MaskMod,
    causal_mask,
    document_mask,
    flash_attn,
    flash_attn_varlen,
    reference_attn,
    reference_attn_varlen,
)

cuda = pytest.mark.cuda

DTYPES = [jnp.bfloat16, jnp.float16]


def cu_from(lens):
    return jnp.asarray(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)


def pack(lens_q, lens_k=None, h_q=2, h_kv=None, d=64, d_v=None, dtype=jnp.float32, seed=0):
    """Packed q/k/v plus their cu_seqlens; lens_k=None shares lens_q."""
    lens_k = lens_q if lens_k is None else lens_k
    h_kv = h_q if h_kv is None else h_kv
    d_v = d if d_v is None else d_v
    rng = np.random.default_rng(seed)
    mk = lambda *s: jnp.asarray(rng.standard_normal(s), dtype)
    q = mk(sum(lens_q), h_q, d)
    k = mk(sum(lens_k), h_kv, d)
    v = mk(sum(lens_k), h_kv, d_v)
    return q, k, v, cu_from(lens_q), cu_from(lens_k)


def loop_oracle(q, k, v, cu_q, cu_k, mask_for=None, **kw):
    """Per-sequence loop over the dense reference: the ground-truth anchor.

    ``mask_for(b)`` supplies a per-sequence mask_mod instance (the dense
    reference sees each sequence as batch 0, so per-``b`` array fields must
    be sliced by the caller).
    """
    h, d_v = q.shape[1], v.shape[2]
    cu_q_, cu_k_ = np.asarray(cu_q), np.asarray(cu_k)
    outs, lses = [], []
    for b in range(len(cu_q_) - 1):
        qs = q[cu_q_[b] : cu_q_[b + 1]][None]
        ks = k[cu_k_[b] : cu_k_[b + 1]][None]
        vs = v[cu_k_[b] : cu_k_[b + 1]][None]
        if qs.shape[1] == 0:
            continue
        if ks.shape[1] == 0:  # no visible key: zeros out, -inf lse
            outs.append(jnp.zeros((qs.shape[1], h, d_v), q.dtype))
            lses.append(jnp.full((h, qs.shape[1]), -jnp.inf, jnp.float32))
            continue
        kw_b = dict(kw)
        if mask_for is not None:
            kw_b["mask_mod"] = mask_for(b)
        out, lse = reference_attn(qs, ks, vs, return_lse=True, **kw_b)
        outs.append(out[0])
        lses.append(lse[0])
    return jnp.concatenate(outs, axis=0), jnp.concatenate(lses, axis=-1)


# (id, lens_q, lens_k): tile-aligned, ragged tails, tiny, zero-length
# sequences on both sides, decode (one query against long caches), s_q >
# s_kv per sequence, a single sequence, many small ones.
LENSETS = [
    ("aligned", [128, 128], None),
    ("ragged", [37, 71, 160], None),
    ("tiny", [1, 2, 3], None),
    ("zeros", [0, 70, 30, 50], [25, 40, 0, 30]),
    ("decode", [1] * 6, [80, 33, 128, 1, 64, 256]),
    ("qlong", [64, 128], [16, 48]),
    ("single", [333], None),
    ("manysmall", [7] * 19, None),
]
LENS_IDS = [t[0] for t in LENSETS]
LENS_ARGS = [(t[1], t[2]) for t in LENSETS]

MASKING = [dict(), dict(causal=True), dict(window_size=(24, 8))]


# ---------------------------------------------------------------- reference semantics


@pytest.mark.parametrize("kw", MASKING, ids=["plain", "causal", "window"])
@pytest.mark.parametrize("lens_q,lens_k", LENS_ARGS, ids=LENS_IDS)
def test_reference_matches_loop(lens_q, lens_k, kw):
    """The packed reference equals the per-sequence loop: segment isolation
    plus per-sequence bottom-right alignment of causal/window."""
    q, k, v, cu_q, cu_k = pack(lens_q, lens_k)
    out, lse = reference_attn_varlen(q, k, v, cu_q, cu_k, return_lse=True, **kw)
    want, want_lse = loop_oracle(q, k, v, cu_q, cu_k, **kw)
    assert_close(out, want, dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(lse), np.asarray(want_lse), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("gqa,d_v", [(2, None), (4, 32)])
def test_reference_gqa_and_mixed_dims(gqa, d_v):
    q, k, v, cu_q, cu_k = pack([37, 71, 26], h_q=2 * gqa, h_kv=2, d_v=d_v)
    out = reference_attn_varlen(q, k, v, cu_q, cu_k, causal=True)
    want, _ = loop_oracle(q, k, v, cu_q, cu_k, causal=True)
    assert_close(out, want, dtype=jnp.float32)


@dataclass(frozen=True)
class _PrefixMask(MaskMod):
    """Per-sequence prefix: array field indexed by the sequence index."""

    prefix_len: jax.Array  # (batch,)

    def __call__(self, b, h, q, kv):
        return kv < self.prefix_len[b]


def test_reference_mask_mod_sees_local_indices():
    """mask_mod gets the sequence index as b and sequence-local q/kv indices;
    checked against per-sequence masks sliced for the loop."""
    lens = [40, 64, 24]
    q, k, v, cu_q, cu_k = pack(lens)
    rng = np.random.default_rng(3)
    ids = jnp.asarray(rng.integers(0, 3, size=(3, max(lens))), jnp.int32)
    prefix = jnp.asarray([10, 50, 5], jnp.int32)
    for mk_full, mk_seq in (
        (document_mask(ids), lambda b: document_mask(ids[b : b + 1])),
        (_PrefixMask(prefix), lambda b: _PrefixMask(prefix[b : b + 1])),
        (causal_mask(), lambda b: causal_mask()),
    ):
        out = reference_attn_varlen(q, k, v, cu_q, cu_k, mask_mod=mk_full)
        want, _ = loop_oracle(q, k, v, cu_q, cu_k, mask_for=mk_seq)
        assert_close(out, want, dtype=jnp.float32, msg=repr(mk_full))


def test_reference_empty_rows_convention():
    """A sequence with keys but no visible ones (len_k = 0): zero output,
    -inf lse — and everything stays finite."""
    q, k, v, cu_q, cu_k = pack([16, 20, 8], [16, 0, 8])
    out, lse = reference_attn_varlen(q, k, v, cu_q, cu_k, return_lse=True)
    out, lse = np.asarray(out), np.asarray(lse)
    assert (out[16:36] == 0).all()
    assert (lse[:, 16:36] == -np.inf).all()
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- validation


def test_rejects_bad_ranks():
    q, k, v, cu_q, cu_k = pack([16, 16])
    with pytest.raises(ValueError, match="packed"):
        flash_attn_varlen(q[None], k, v, cu_q, cu_k)
    with pytest.raises(ValueError, match="1D"):
        flash_attn_varlen(q, k, v, cu_q[None], cu_k)
    with pytest.raises(ValueError, match="1D"):
        flash_attn_varlen(q, k, v, jnp.zeros((1,), jnp.int32), cu_k)


def test_rejects_non_integer_cu():
    q, k, v, cu_q, cu_k = pack([16, 16])
    with pytest.raises(TypeError, match="integer"):
        flash_attn_varlen(q, k, v, cu_q.astype(jnp.float32), cu_k)


def test_rejects_cu_batch_mismatch():
    q, k, v, cu_q, _cu_k = pack([16, 16])
    with pytest.raises(ValueError, match="batch mismatch"):
        flash_attn_varlen(q, k, v, cu_q, cu_from([10, 10, 12]))


def test_rejects_qkv_disagreement():
    q, k, v, cu_q, cu_k = pack([16, 16])
    with pytest.raises(ValueError, match="dtypes"):
        flash_attn_varlen(q, k.astype(jnp.float16), v.astype(jnp.float16), cu_q, cu_k)
    with pytest.raises(ValueError, match="head_dim mismatch"):
        flash_attn_varlen(q[:, :, :32], k, v, cu_q, cu_k)
    with pytest.raises(ValueError, match="multiple"):
        flash_attn_varlen(jnp.concatenate([q, q[:, :1]], axis=1), k, v, cu_q, cu_k)


def test_vmap_rejected():
    q, k, v, cu_q, cu_k = pack([16, 16])
    f = lambda q: flash_attn_varlen(q, k, v, cu_q, cu_k, backend="reference")
    with pytest.raises(NotImplementedError, match="vmap"):
        jax.vmap(f)(jnp.stack([q, q]))


def test_public_surface():
    assert {"flash_attn_varlen", "reference_attn_varlen"} <= set(fa4_jax.__all__)


# ---------------------------------------------------------------- wiring vs plain autodiff
#
# On the reference backend the primitives' JVP/transpose rules must reproduce
# ordinary JAX autodiff of the (plain-jnp) varlen reference — this checks the
# cu_seqlens operand plumbing through all three primitives.


def _wiring_pair(**kw):
    q, k, v, cu_q, cu_k = pack([37, 71, 26], seed=11)
    rng = np.random.default_rng(12)
    cot = jnp.asarray(rng.standard_normal((q.shape[0], q.shape[1], v.shape[2])), jnp.float32)
    prim = lambda q, k, v: (
        flash_attn_varlen(q, k, v, cu_q, cu_k, backend="reference", **kw) * cot
    ).sum()
    direct = lambda q, k, v: (reference_attn_varlen(q, k, v, cu_q, cu_k, **kw) * cot).sum()
    return (q, k, v), prim, direct


@pytest.mark.parametrize("kw", MASKING, ids=["plain", "causal", "window"])
def test_wiring_grad_and_jvp(kw):
    args, prim, direct = _wiring_pair(**kw)
    got = jax.grad(prim, argnums=(0, 1, 2))(*args)
    want = jax.grad(direct, argnums=(0, 1, 2))(*args)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=jnp.float32, msg=f"d{name}")

    rng = np.random.default_rng(13)
    tangents = tuple(jnp.asarray(rng.standard_normal(a.shape), a.dtype) for a in args)
    _, t_got = jax.jvp(prim, args, tangents)
    _, t_want = jax.jvp(direct, args, tangents)
    assert_close(t_got, t_want, dtype=jnp.float32)


def test_wiring_lse_is_differentiable():
    q, k, v, cu_q, cu_k = pack([30, 50], seed=14)
    f = lambda be: lambda q: flash_attn_varlen(
        q, k, v, cu_q, cu_k, causal=True, return_lse=True, backend=be
    )[1].sum()
    got = jax.grad(f("reference"))(q)
    want = jax.grad(lambda q: reference_attn_varlen(q, k, v, cu_q, cu_k, causal=True, return_lse=True)[1].sum())(q)
    assert_close(got, want, dtype=jnp.float32)


# ---------------------------------------------------------------- kernels


def _both(q, k, v, cu_q, cu_k, **kw):
    got = flash_attn_varlen(q, k, v, cu_q, cu_k, backend="cute", **kw)
    want = flash_attn_varlen(q, k, v, cu_q, cu_k, backend="reference", **kw)
    return got, want


def _grads(q, k, v, cu_q, cu_k, cot=None, **kw):
    def loss(backend):
        def f(q, k, v):
            out = flash_attn_varlen(q, k, v, cu_q, cu_k, backend=backend, **kw)
            return out.sum() if cot is None else (out * cot).sum()

        return f

    got = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    return got, want


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens_q,lens_k", LENS_ARGS, ids=LENS_IDS)
def test_fwd(lens_q, lens_k, causal, dtype):
    q, k, v, cu_q, cu_k = pack(lens_q, lens_k, dtype=dtype)
    got, want = _both(q, k, v, cu_q, cu_k, causal=causal)
    assert_close(got, want, dtype=dtype)


@cuda
@pytest.mark.parametrize("window", [(32, 32), (16, 0), (0, 16), (None, 5), (3, None)])
def test_fwd_window(window):
    q, k, v, cu_q, cu_k = pack([37, 200, 71], dtype=jnp.bfloat16)
    got, want = _both(q, k, v, cu_q, cu_k, window_size=window)
    assert_close(got, want, dtype=q.dtype)


@cuda
@pytest.mark.parametrize("kw", MASKING, ids=["plain", "causal", "window"])
@pytest.mark.parametrize("lens_q,lens_k", LENS_ARGS[:5], ids=LENS_IDS[:5])
def test_fwd_lse(lens_q, lens_k, kw):
    """lse at 1e-3 is the sharpest edge detector: one dropped or leaked key
    across a sequence boundary shifts it by ~1/s."""
    q, k, v, cu_q, cu_k = pack(lens_q, lens_k, dtype=jnp.bfloat16)
    got, lse_got = flash_attn_varlen(q, k, v, cu_q, cu_k, return_lse=True, backend="cute", **kw)
    want, lse_want = flash_attn_varlen(q, k, v, cu_q, cu_k, return_lse=True, backend="reference", **kw)
    assert_close(got, want, dtype=q.dtype)
    assert_close(lse_got, lse_want, atol=1e-3, rtol=1e-3)


@cuda
@pytest.mark.parametrize(
    "h_kv,gqa,d,d_v",
    [
        (2, 2, 64, None),  # GQA, equal dims
        (1, 4, 64, None),  # GQA ratio 4
        (2, 1, 128, None),  # head_dim 128
        (2, 1, 96, None),  # padded head dim
        (2, 1, 64, 32),  # mixed head dims
        (2, 2, 64, 32),  # GQA + mixed dims
    ],
)
def test_fwd_head_configs(h_kv, gqa, d, d_v):
    q, k, v, cu_q, cu_k = pack([37, 71, 160], h_q=h_kv * gqa, h_kv=h_kv, d=d, d_v=d_v, dtype=jnp.bfloat16)
    got, want = _both(q, k, v, cu_q, cu_k, causal=True)
    assert_close(got, want, dtype=q.dtype)


@cuda
def test_fwd_mask_mod_local_indices():
    """Document + prefix masks on the kernels: the predicate must see the
    sequence index and sequence-local coordinates."""
    lens = [40, 64, 24]
    q, k, v, cu_q, cu_k = pack(lens, dtype=jnp.bfloat16)
    rng = np.random.default_rng(3)
    ids = jnp.asarray(rng.integers(0, 3, size=(3, max(lens))), jnp.int32)
    prefix = jnp.asarray([10, 50, 5], jnp.int32)
    for mm in (document_mask(ids), _PrefixMask(prefix)):
        got, want = _both(q, k, v, cu_q, cu_k, mask_mod=mm)
        assert_close(got, want, dtype=q.dtype, msg=repr(mm))


@cuda
@pytest.mark.parametrize("native", [dict(causal=True), dict(window_size=(48, 16))])
def test_fwd_native_mask_combines_with_mask_mod(native):
    q, k, v, cu_q, cu_k = pack([80, 120], dtype=jnp.bfloat16)
    ids = jnp.asarray(np.repeat(np.arange(4), 30)[None].repeat(2, 0), jnp.int32)
    got, want = _both(q, k, v, cu_q, cu_k, mask_mod=document_mask(ids), **native)
    assert_close(got, want, dtype=q.dtype)


@cuda
def test_single_sequence_matches_dense():
    """cu = [0, total] must agree with the dense kernel on the same data."""
    s = 200
    q, k, v, cu_q, cu_k = pack([s], dtype=jnp.bfloat16)
    got = flash_attn_varlen(q, k, v, cu_q, cu_k, causal=True, backend="cute")
    dense = flash_attn(q[None], k[None], v[None], causal=True, backend="cute")[0]
    assert_close(got, dense, dtype=q.dtype)


# ---------------------------------------------------------------- gradients


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens_q,lens_k", LENS_ARGS, ids=LENS_IDS)
def test_grad(lens_q, lens_k, causal, dtype):
    q, k, v, cu_q, cu_k = pack(lens_q, lens_k, dtype=dtype)
    rng = np.random.default_rng(1)
    cot = jnp.asarray(rng.standard_normal((q.shape[0], q.shape[1], v.shape[2])), dtype)
    got, want = _grads(q, k, v, cu_q, cu_k, cot, causal=causal)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=dtype, msg=f"d{name}")


@cuda
@pytest.mark.parametrize("window", [(32, 32), (16, 0), (None, 5)])
def test_grad_window(window):
    q, k, v, cu_q, cu_k = pack([37, 200, 71], dtype=jnp.bfloat16)
    got, want = _grads(q, k, v, cu_q, cu_k, window_size=window)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=q.dtype, msg=f"d{name}")


@cuda
@pytest.mark.parametrize(
    "h_kv,gqa,d,d_v",
    [
        (2, 2, 64, None),  # GQA: fp32 dK/dV accumulators + postprocess
        (1, 8, 64, None),  # GQA ratio 8
        (2, 2, 64, 32),  # GQA + mixed dims: SM80-MMA fallback on sm_100
        (2, 1, 64, 32),  # non-GQA mixed dims: SM80-MMA fallback, direct dK/dV
        (2, 1, 128, None),  # head_dim 128
    ],
)
@pytest.mark.parametrize("lens", [[128, 128], [37, 71, 160]], ids=["aligned", "ragged"])
def test_grad_head_configs(lens, h_kv, gqa, d, d_v):
    """Ragged lens exercise the tile-rounded accumulator offsets (regression:
    upstream's SM80 GQA varlen path used unrounded offsets, see VENDOR.md)."""
    q, k, v, cu_q, cu_k = pack(lens, h_q=h_kv * gqa, h_kv=h_kv, d=d, d_v=d_v, dtype=jnp.bfloat16)
    rng = np.random.default_rng(2)
    cot = jnp.asarray(rng.standard_normal((q.shape[0], q.shape[1], v.shape[2])), jnp.bfloat16)
    got, want = _grads(q, k, v, cu_q, cu_k, cot, causal=True)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=q.dtype, msg=f"d{name}")


@cuda
def test_grad_mask_mod():
    lens = [40, 64, 24]
    q, k, v, cu_q, cu_k = pack(lens, dtype=jnp.bfloat16)
    prefix = jnp.asarray([10, 50, 5], jnp.int32)
    got, want = _grads(q, k, v, cu_q, cu_k, mask_mod=_PrefixMask(prefix))
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=q.dtype, msg=f"d{name}")


@cuda
def test_grad_zero_length_and_empty_rows_finite():
    """Zero-length sequences on both sides plus a keyless sequence: all
    gradients finite, and dq of empty rows exactly zero."""
    q, k, v, cu_q, cu_k = pack([0, 70, 30, 50], [25, 40, 0, 30], dtype=jnp.bfloat16)
    got, want = _grads(q, k, v, cu_q, cu_k, causal=True)
    for g, w, name in zip(got, want, "qkv"):
        assert np.isfinite(np.asarray(g, np.float32)).all(), name
        assert_close(g, w, dtype=q.dtype, msg=f"d{name}")
    assert (np.asarray(got[0], np.float32)[70:100] == 0).all()  # seq 2: len_k == 0


@cuda
def test_grad_with_unused_lse():
    q, k, v, cu_q, cu_k = pack([37, 91], dtype=jnp.bfloat16)
    loss = lambda be: lambda q: flash_attn_varlen(
        q, k, v, cu_q, cu_k, causal=True, return_lse=True, backend=be
    )[0].sum()
    got = jax.grad(loss("cute"))(q)
    want = jax.grad(loss("reference"))(q)
    assert_close(got, want, dtype=q.dtype)


# ---------------------------------------------------------------- jvp


@cuda
def test_jvp_raises_on_cute_works_on_reference():
    q, k, v, cu_q, cu_k = pack([37, 91], dtype=jnp.bfloat16)
    rng = np.random.default_rng(5)
    tangents = tuple(jnp.asarray(rng.standard_normal(t.shape), t.dtype) for t in (q, k, v))
    f = lambda be: lambda q, k, v: flash_attn_varlen(q, k, v, cu_q, cu_k, backend=be)
    with pytest.raises(NotImplementedError, match="varlen"):
        jax.jvp(f("cute"), (q, k, v), tangents)
    _, t_ref = jax.jvp(f("reference"), (q, k, v), tangents)
    assert np.isfinite(np.asarray(t_ref, np.float32)).all()


# ---------------------------------------------------------------- jit / caching


@cuda
def test_jit_matches_eager():
    q, k, v, cu_q, cu_k = pack([37, 91], dtype=jnp.bfloat16)
    f = lambda q, k, v: flash_attn_varlen(q, k, v, cu_q, cu_k, causal=True, backend="cute")
    loss = lambda q, k, v: f(q, k, v).sum()
    for a, b in zip([f(q, k, v)], [jax.jit(f)(q, k, v)]):
        assert_close(a, b, atol=0, rtol=0)
    eager = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    jitted = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)
    for a, b in zip(eager, jitted):
        assert_close(a, b, atol=0, rtol=0)


@cuda
def test_new_packing_reuses_kernels():
    """Different cu contents, different total length, different batch size —
    all symbolic, so nothing recompiles after the first call."""
    import cutejax

    def run(lens):
        q, k, v, cu_q, cu_k = pack(lens, dtype=jnp.bfloat16)
        out = flash_attn_varlen(q, k, v, cu_q, cu_k, causal=True, backend="cute")
        return jax.grad(lambda q: flash_attn_varlen(q, k, v, cu_q, cu_k, causal=True, backend="cute").sum())(q), out

    run([64, 64])
    before = cutejax.cache_info()["size"]
    run([37, 91])  # same total, different split
    run([50, 60, 70, 33])  # different total AND batch size
    assert cutejax.cache_info()["size"] == before, "varlen packing must not recompile"
