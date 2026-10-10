"""Cheap API-level checks: input validation, window canonicalization, and the
MaskMod pytree contract (what keys a kernel vs. what is runtime data). These
run on the cute backend where a kernel is involved; the rest never reaches a
kernel at all."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_close, random_qkv

from fa4_jax import (
    FnMask,
    MaskMod,
    and_masks,
    causal_mask,
    document_mask,
    flash_attn,
    or_masks,
    reference_attn,
    sliding_window_mask,
    static,
)
from fa4_jax.attention import _resolve_causal_local_window
from fa4_jax.reference import dense_mask


def _qkv(dtype=jnp.bfloat16):
    return random_qkv(np.random.default_rng(0), 1, 64, 64, 2, 2, 64, dtype=dtype)


# ---------------------------------------------------------------- validation


def test_rejects_bad_rank():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="batch, seqlen, heads, head_dim"):
        flash_attn(q[0], k, v)


def test_rejects_dtype_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="dtypes must match"):
        flash_attn(q, k.astype(jnp.float16), v)


def test_rejects_bad_gqa_ratio():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="not a multiple"):
        flash_attn(q[:, :, :1].repeat(3, 2), k, v)


def test_rejects_kv_shape_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="k/v"):
        flash_attn(q, k, v[:, :4])


def test_rejects_head_dim_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="head_dim mismatch"):
        flash_attn(q[..., :32], k, v)


def test_rejects_raw_callable_mask():
    q, k, v = _qkv()
    with pytest.raises(TypeError, match="MaskMod"):
        flash_attn(q, k, v, mask_mod=lambda b, h, qi, ki: ki <= qi)


@pytest.mark.cuda
def test_rejects_f32_on_cute():
    q, k, v = _qkv(jnp.float32)
    with pytest.raises(ValueError, match="fp16/bf16"):
        flash_attn(q, k, v, backend="cute")


@pytest.mark.cuda
def test_unsupported_head_dim_never_poisons_context():
    """head_dim 256: whether it fits is family- and device-dependent (the
    sm80 family and sm120's smaller tiles accommodate it; the tcgen05
    wrapper rejects it). The invariant is that flash_attn either raises
    a clean ValueError or computes correctly — never a failed launch that
    poisons the CUDA context."""
    q, k, v = random_qkv(np.random.default_rng(0), 1, 64, 64, 1, 1, 256, dtype=jnp.bfloat16)
    try:
        got = flash_attn(q, k, v, backend="cute")
    except ValueError as e:
        assert "cannot run head_dim=256" in str(e)
    else:
        want = flash_attn(q, k, v, backend="reference")
        assert_close(got, want, dtype=q.dtype)
    q, k, v = random_qkv(np.random.default_rng(0), 1, 64, 64, 1, 1, 20, dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="multiples of 8"):
        flash_attn(q, k, v, backend="cute")


def test_unknown_backend():
    q, k, v = _qkv()
    with pytest.raises(KeyError):
        flash_attn(q, k, v, backend="nope")


def test_default_backend_follows_jax_device():
    from fa4_jax.attention import _default_backend

    want = "cute" if jax.default_backend() == "gpu" else "reference"
    assert _default_backend() == want


def test_window_canonicalization():
    # bare right-window 0 is the causal kernel
    assert _resolve_causal_local_window(False, None, 0) == (True, False, None, None)
    # causal flag forces right=0 and stays causal
    assert _resolve_causal_local_window(True, None, None) == (True, False, None, None)
    # causal + left window becomes the local kernel with right=0
    assert _resolve_causal_local_window(True, 5, None) == (False, True, 5, 0)
    # a window that excludes everything is dropped entirely (upstream rule)
    assert _resolve_causal_local_window(False, 2, -5) == (False, False, None, None)
    # plain two-sided window
    assert _resolve_causal_local_window(False, 3, 4) == (False, True, 3, 4)


def test_causal_flag_equals_right_zero_window():
    q, k, v = _qkv()
    a = flash_attn(q, k, v, causal=True, backend="reference")
    b = flash_attn(q, k, v, window_size=(None, 0), backend="reference")
    assert_close(a, b, atol=0, rtol=0)


def test_causal_and_window_combine():
    q, k, v = _qkv()
    a = flash_attn(q, k, v, causal=True, window_size=(3, None), backend="reference")
    b = flash_attn(q, k, v, window_size=(3, 0), backend="reference")
    assert_close(a, b, atol=0, rtol=0)


# ---------------------------------------------------------------- mask pytrees


def _dense(mask, b=2, h=3, s_q=7, s_kv=9):
    return np.asarray(dense_mask(b, h, s_q, s_kv, False, (None, None), mask))


def test_dense_mask_semantics():
    q, kv = np.mgrid[0:7, 0:9]
    np.testing.assert_array_equal(
        _dense(causal_mask()), np.broadcast_to(kv <= q, (2, 3, 7, 9))
    )
    want = (q - kv <= 2) & (kv - q <= 1)
    np.testing.assert_array_equal(
        _dense(sliding_window_mask(2, 1)), np.broadcast_to(want, (2, 3, 7, 9))
    )
    a, b = causal_mask(), sliding_window_mask(2, None)
    np.testing.assert_array_equal(_dense(and_masks(a, b)), _dense(a) & _dense(b))
    np.testing.assert_array_equal(_dense(or_masks(a, b)), _dense(a) | _dense(b))
    head = FnMask(lambda b, h, q, kv: (kv <= q) | (h == 0), label="h0-full")
    d = _dense(head)
    assert d[:, 0].all()
    np.testing.assert_array_equal(d[:, 1:], np.broadcast_to(kv <= q, (2, 2, 7, 9)))


def test_native_masks_are_bottom_right_aligned():
    """causal/window_size line the last query up with the last key when
    s_q != s_kv (flash-attention convention); mask mods see raw indices."""
    q, kv = np.mgrid[0:4, 0:10]
    off = 10 - 4
    got = np.asarray(dense_mask(1, 1, 4, 10, True, (None, None), None))[0, 0]
    np.testing.assert_array_equal(got, kv <= q + off)
    assert got[-1].all() and got[0].sum() == off + 1  # row 0 sees keys 0..6
    got = np.asarray(dense_mask(1, 1, 4, 10, False, (2, 1), None))[0, 0]
    np.testing.assert_array_equal(got, (q + off - kv <= 2) & (kv - q - off <= 1))
    # s_q > s_kv: early queries see nothing under causal
    got = np.asarray(dense_mask(1, 1, 10, 4, True, (None, None), None))[0, 0]
    q, kv = np.mgrid[0:10, 0:4]
    np.testing.assert_array_equal(got, kv <= q - 6)
    assert not got[:6].any()
    # the mask-mod causal is top-left, so it differs from causal=True here
    mm = np.asarray(dense_mask(1, 1, 4, 10, False, (None, None), causal_mask()))[0, 0]
    q, kv = np.mgrid[0:4, 0:10]
    np.testing.assert_array_equal(mm, kv <= q)


def test_document_mask_dense():
    ids = jnp.asarray([[0, 0, 1, 1, 1, 2, 2, 2, 2]])
    m = _dense(document_mask(ids), b=1, s_q=9, s_kv=9)
    want = np.asarray(ids[0])[:, None] == np.asarray(ids[0])[None, :]
    np.testing.assert_array_equal(m, np.broadcast_to(want, (1, 3, 9, 9)))


def test_static_fields_key_the_treedef():
    """Static fields distinguish kernels; array fields don't; the treedef
    round-trips to a working callable (the path the cute backend takes)."""
    a, b, c = (
        sliding_window_mask(2, 1),
        sliding_window_mask(2, 1),
        sliding_window_mask(3, 1),
    )
    assert jax.tree.structure(a) == jax.tree.structure(b)
    assert jax.tree.structure(a) != jax.tree.structure(c)
    assert jax.tree.leaves(a) == []

    ids1 = jnp.asarray([[0, 0, 1]])
    ids2 = jnp.asarray([[0, 1, 1]])
    assert jax.tree.structure(document_mask(ids1)) == jax.tree.structure(
        document_mask(ids2)
    )

    full = and_masks(causal_mask(), document_mask(ids1))
    leaves, treedef = jax.tree.flatten(full)
    assert len(leaves) == 1
    rebuilt = treedef.unflatten(leaves)
    np.testing.assert_array_equal(
        _dense(rebuilt, b=1, s_q=3, s_kv=3), _dense(full, b=1, s_q=3, s_kv=3)
    )


@pytest.mark.cuda
def test_custom_mask_class():
    """User-defined MaskMod subclasses register as pytrees automatically and
    run on the kernels."""

    @dataclass(frozen=True)
    class PrefixMask(MaskMod):
        prefix_len: jax.Array  # (batch,) runtime
        stride: int = static(default=1)

        def __call__(self, b, h, q, kv):
            return (kv < self.prefix_len[b]) | ((kv <= q) & (kv % self.stride == 0))

    m = PrefixMask(jnp.asarray([2, 4]), stride=2)
    assert jax.tree.leaves(m) == [m.prefix_len]
    d = _dense(m, b=2, h=1, s_q=6, s_kv=6)
    q, kv = np.mgrid[0:6, 0:6]
    for bi, p in enumerate([2, 4]):
        np.testing.assert_array_equal(d[bi, 0], (kv < p) | ((kv <= q) & (kv % 2 == 0)))

    q, k, v = random_qkv(
        np.random.default_rng(1), 2, 96, 96, 1, 1, 64, dtype=jnp.bfloat16
    )
    m = PrefixMask(jnp.asarray([16, 40], jnp.int32), stride=3)
    got = flash_attn(q, k, v, mask_mod=m, backend="cute")
    want = flash_attn(q, k, v, mask_mod=m, backend="reference")
    assert_close(got, want, dtype=q.dtype)


def test_mask_constructor_validation():
    with pytest.raises(ValueError, match="unbounded"):
        sliding_window_mask(None, None)
    with pytest.raises(ValueError, match="at least one"):
        and_masks()
    with pytest.raises(ValueError, match="at least one"):
        or_masks()


def test_non_array_leaf_rejected():
    @dataclass(frozen=True)
    class Bad(MaskMod):
        left: int  # forgot static()

        def __call__(self, b, h, q, kv):
            return q - kv <= self.left

    q, k, v = _qkv()
    with pytest.raises(TypeError, match="static"):
        flash_attn(q, k, v, mask_mod=Bad(3))


def test_mask_arrays_not_differentiable():
    q, k, v = _qkv()
    ids = jnp.zeros((1, 64), jnp.float32)  # float so jax will try to differentiate
    f = lambda ids: flash_attn(q, k, v, mask_mod=document_mask(ids)).sum()
    with pytest.raises(NotImplementedError, match="aux"):
        jax.grad(f)(ids)


# ---------------------------------------------------------------- wiring vs plain autodiff
#
# flash_attn(backend="reference") runs the custom primitives (JVP rule,
# tangent primitive, transpose rule) over the reference backend; reference_attn
# is the same math under ordinary jax autodiff. Agreement pins the wiring.


def _wiring_case(seed=7, b=2, s=24, h_q=2, h_kv=2, d=16):
    rng = np.random.default_rng(seed)
    q, k, v = random_qkv(rng, b, s, s, h_q, h_kv, d)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), t.dtype) for t in (q, k, v))
    return q, k, v, dq, dk, dv


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(causal=True),
        dict(window_size=(4, 2)),
        dict(mask_mod=sliding_window_mask(4, 2)),
    ],
)
def test_wiring_grad_and_jvp(kw):
    q, k, v, dq, dk, dv = _wiring_case()
    f = lambda *a: flash_attn(*a, backend="reference", **kw)
    g = lambda *a: reference_attn(*a, **kw)
    _, tan_f = jax.jvp(f, (q, k, v), (dq, dk, dv))
    _, tan_g = jax.jvp(g, (q, k, v), (dq, dk, dv))
    assert_close(tan_f, tan_g, dtype=q.dtype)
    gf = jax.grad(lambda *a: f(*a).sum(), argnums=(0, 1, 2))(q, k, v)
    gg = jax.grad(lambda *a: g(*a).sum(), argnums=(0, 1, 2))(q, k, v)
    for got, want in zip(gf, gg):
        assert_close(got, want, dtype=q.dtype)


def test_wiring_lse_is_differentiable():
    """return_lse=True: grad and jvp flow through lse (t_lse and the d_lse
    cotangent path of the transpose rule)."""
    q, k, v, dq, dk, dv = _wiring_case(seed=8)

    def f(q, k, v):
        out, lse = flash_attn(q, k, v, return_lse=True, backend="reference")
        return out.sum() + (lse * 0.7).sum()

    def g(q, k, v):
        out, lse = reference_attn(q, k, v, return_lse=True)
        return out.sum() + (lse * 0.7).sum()

    _, tan_f = jax.jvp(f, (q, k, v), (dq, dk, dv))
    _, tan_g = jax.jvp(g, (q, k, v), (dq, dk, dv))
    assert_close(tan_f, tan_g, atol=1e-3, rtol=1e-3)
    gf = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    gg = jax.grad(g, argnums=(0, 1, 2))(q, k, v)
    for got, want in zip(gf, gg):
        assert_close(got, want, dtype=q.dtype)


def test_wiring_partial_tangents():
    """Differentiating w.r.t. a subset of (q, k, v): the JVP rule instantiates
    the symbolic-zero tangents of the rest (ad.Zero -> zeros)."""
    q, k, v, dq, dk, dv = _wiring_case(seed=11)
    f = lambda q, k, v: flash_attn(q, k, v, causal=True, backend="reference")
    g = lambda q, k, v: reference_attn(q, k, v, causal=True)
    for argnum, tangent in ((0, dq), (1, dk), (2, dv)):
        args = [q, k, v]
        ff = lambda x, i=argnum: f(*[x if j == i else a for j, a in enumerate(args)])
        gg = lambda x, i=argnum: g(*[x if j == i else a for j, a in enumerate(args)])
        _, tf = jax.jvp(ff, (args[argnum],), (tangent,))
        _, tg = jax.jvp(gg, (args[argnum],), (tangent,))
        assert_close(tf, tg, dtype=q.dtype, msg=f"jvp argnum {argnum}")
        gf = jax.grad(lambda x: ff(x).sum())(args[argnum])
        gg_ = jax.grad(lambda x: gg(x).sum())(args[argnum])
        assert_close(gf, gg_, dtype=q.dtype, msg=f"grad argnum {argnum}")


def test_wiring_grad_with_unused_lse():
    """return_lse=True but only `out` is differentiated: the lse cotangent is
    a symbolic zero and the transpose must not pass a d_lse."""
    q, k, v, *_ = _wiring_case(seed=12)
    f = lambda q, k, v: flash_attn(q, k, v, return_lse=True, backend="reference")[0].sum()
    g = lambda q, k, v: reference_attn(q, k, v).sum()
    gf = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    gg = jax.grad(g, argnums=(0, 1, 2))(q, k, v)
    for got, want in zip(gf, gg):
        assert_close(got, want, dtype=q.dtype)


def test_wiring_gqa_grad():
    q, k, v, *_ = _wiring_case(seed=9, h_q=4, h_kv=2)
    gf = jax.grad(
        lambda *a: flash_attn(*a, backend="reference").sum(), argnums=(0, 1, 2)
    )(q, k, v)
    gg = jax.grad(lambda *a: reference_attn(*a).sum(), argnums=(0, 1, 2))(q, k, v)
    for got, want in zip(gf, gg):
        assert_close(got, want, dtype=q.dtype)


def test_linearize_transpose_roundtrip():
    """<J u, w> == <u, J^T w>: the transpose rule is the true adjoint of the
    tangent map (dot-product test, independent of any reference)."""
    q, k, v, *u = _wiring_case(seed=13, s=17)
    f = lambda q, k, v: flash_attn(q, k, v, backend="reference")
    out, f_jvp = jax.linearize(f, q, k, v)
    w = jnp.asarray(np.random.default_rng(15).standard_normal(out.shape), out.dtype)
    Ju = f_jvp(*u)
    JTw = jax.linear_transpose(f_jvp, q, k, v)(w)
    lhs = jnp.vdot(Ju, w)
    rhs = sum(jnp.vdot(a, b) for a, b in zip(u, JTw))
    assert_close(lhs, rhs, atol=1e-3, rtol=1e-3)


def test_empty_rows_convention():
    """No visible key -> out = 0, lse = -inf, zero gradient; never NaN."""
    q, k, v, *_ = _wiring_case(seed=10, s=8)
    out, lse = reference_attn(
        q, k, v, window_size=(None, -3), return_lse=True
    )  # q < 3 see nothing
    assert np.isfinite(np.asarray(out)).all()
    np.testing.assert_array_equal(np.asarray(out[:, :3]), 0)
    assert np.isneginf(np.asarray(lse[:, :, :3])).all()
    assert np.isfinite(np.asarray(lse[:, :, 3:])).all()
    g = jax.grad(lambda q: reference_attn(q, k, v, window_size=(None, -3)).sum())(q)
    assert np.isfinite(np.asarray(g)).all()


# ---------------------------------------------------------------- vmap
#
# Without a mask the batching rule folds the vmapped axis into batch and
# rebinds the primitives (so it runs on the kernels and keeps the custom
# differentiation rules). With a mask it vmaps the backend impl instead, which
# works on the reference backend only — see docs/mask-axes-vmap.md.


def _vmapped(n=3, b=2, s=64, h=2, d=64, seed=21):
    rng = np.random.default_rng(seed)
    mk = lambda: jnp.asarray(rng.standard_normal((n, b, s, h, d)), jnp.bfloat16)
    return mk(), mk(), mk()


def _loop(f, *args, axes):
    n = next(a.shape[0] for a, ax in zip(args, axes) if ax is not None)
    pick = lambda a, ax, i: a if ax is None else a[i]
    return jnp.stack(
        [f(*(pick(a, ax, i) for a, ax in zip(args, axes))) for i in range(n)]
    )


@pytest.mark.cuda
@pytest.mark.parametrize("axes", [(0, 0, 0), (0, None, None), (None, 0, 0)])
@pytest.mark.parametrize("kw", [dict(), dict(causal=True), dict(window_size=(16, 8))])
def test_vmap_fold_fwd(axes, kw):
    q, k, v = _vmapped()
    args = tuple(a if ax is not None else a[0] for a, ax in zip((q, k, v), axes))
    f = lambda q, k, v: flash_attn(q, k, v, backend="cute", **kw)
    got = jax.vmap(f, in_axes=axes)(*args)
    want = _loop(f, *args, axes=axes)
    assert_close(got, want, atol=0, rtol=0)  # same kernel, same numbers


def test_vmap_fold_nonzero_axes_and_gqa():
    """Mapped axis not in front (the moveaxis branch), out_axes != 0, nested
    vmap, and GQA under the fold; reference backend so it runs anywhere."""
    rng = np.random.default_rng(24)
    n, b, s, d = 3, 2, 16, 8
    q = jnp.asarray(rng.standard_normal((b, n, s, 4, d)), jnp.float32)  # mapped axis 1
    k = jnp.asarray(rng.standard_normal((b, s, 2, n, d)), jnp.float32)  # mapped axis 3
    v = jnp.asarray(rng.standard_normal((n, b, s, 2, d)), jnp.float32)  # mapped axis 0
    f = lambda q, k, v: flash_attn(q, k, v, causal=True, backend="reference")
    got = jax.vmap(f, in_axes=(1, 3, 0), out_axes=2)(q, k, v)
    want = jnp.stack(
        [f(q[:, i], k[:, :, :, i], v[i]) for i in range(n)], axis=2
    )
    assert_close(got, want, atol=1e-5, rtol=1e-5)
    # nested vmap over a second fresh axis
    q2 = jnp.stack([q, q * 0.5])
    got = jax.vmap(jax.vmap(f, in_axes=(1, 3, 0)), in_axes=(0, None, None))(q2, k, v)
    want = jnp.stack([jax.vmap(f, in_axes=(1, 3, 0))(qq, k, v) for qq in q2])
    assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_vmap_grad_through_lse():
    """vmap of a grad that flows through lse exercises fa4_vjp's batching rule
    with has_d_lse=True."""
    rng = np.random.default_rng(25)
    mk = lambda: jnp.asarray(rng.standard_normal((3, 1, 16, 2, 8)), jnp.float32)
    q, k, v = mk(), mk(), mk()

    def loss(q, k, v):
        out, lse = flash_attn(q, k, v, return_lse=True, backend="reference")
        return out.sum() + (lse * 0.3).sum()

    def loss_ref(q, k, v):
        out, lse = reference_attn(q, k, v, return_lse=True)
        return out.sum() + (lse * 0.3).sum()

    got = jax.vmap(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)
    want = jax.vmap(jax.grad(loss_ref, argnums=(0, 1, 2)))(q, k, v)
    for a, b in zip(got, want):
        assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
def test_vmap_fold_lse():
    q, k, v = _vmapped()
    f = lambda q, k, v: flash_attn(q, k, v, return_lse=True, backend="cute")[1]
    got = jax.vmap(f)(q, k, v)
    want = _loop(f, q, k, v, axes=(0, 0, 0))
    assert_close(got, want, atol=0, rtol=0)


@pytest.mark.cuda
def test_vmap_fold_grad_and_jvp():
    """vmap of grad / jvp goes through the folded primitives on the kernels."""
    q, k, v = _vmapped()
    dq, dk, dv = _vmapped(seed=22)
    loss = lambda q, k, v: flash_attn(q, k, v, causal=True, backend="cute").sum()
    g = jax.grad(loss, argnums=(0, 1, 2))
    got = jax.vmap(g)(q, k, v)
    want = [_loop(lambda *a, i=i: g(*a)[i], q, k, v, axes=(0, 0, 0)) for i in range(3)]
    for a, b in zip(got, want):
        assert_close(a, b, atol=0, rtol=0)

    f = lambda q, k, v: flash_attn(q, k, v, causal=True, backend="cute")
    tan = lambda q, k, v, dq, dk, dv: jax.jvp(f, (q, k, v), (dq, dk, dv))[1]
    got = jax.vmap(tan)(q, k, v, dq, dk, dv)
    want = _loop(tan, q, k, v, dq, dk, dv, axes=(0,) * 6)
    assert_close(got, want, atol=0, rtol=0)


def test_vmap_mask_reference_fallback():
    """A mask with batched array leaves: vmaps the reference impl."""
    q, k, v = _vmapped()
    ids = jnp.asarray(np.random.default_rng(23).integers(0, 3, q.shape[:3]), jnp.int32)
    f = lambda q, k, v, ids: flash_attn(
        q, k, v, mask_mod=document_mask(ids), backend="reference"
    )
    got = jax.vmap(f)(q, k, v, ids)
    want = _loop(f, q, k, v, ids, axes=(0, 0, 0, 0))
    assert_close(got, want, dtype=q.dtype)
    # leafless mask, also the fallback path today (it has no axis declaration)
    f = lambda q, k, v: flash_attn(
        q, k, v, mask_mod=sliding_window_mask(8, 4), backend="reference"
    )
    assert_close(jax.vmap(f)(q, k, v), _loop(f, q, k, v, axes=(0, 0, 0)), dtype=q.dtype)


@pytest.mark.cuda
def test_vmap_mask_cute_unsupported():
    """Known gap (starchart iwx1ok): the fallback vmaps cutejax's ffi_call,
    which has no batching rule. Applies to leafless masks too (no axis
    declaration yet, see docs/mask-axes-vmap.md)."""
    q, k, v = _vmapped()
    ids = jnp.asarray(np.random.default_rng(23).integers(0, 3, q.shape[:3]), jnp.int32)
    f = lambda q, k, v, ids: flash_attn(
        q, k, v, mask_mod=document_mask(ids), backend="cute"
    )
    with pytest.raises(NotImplementedError, match="vmap"):
        jax.vmap(f)(q, k, v, ids)
    g = lambda q, k, v: flash_attn(q, k, v, mask_mod=sliding_window_mask(8, 4), backend="cute")
    with pytest.raises(NotImplementedError, match="vmap"):
        jax.vmap(g)(q, k, v)
