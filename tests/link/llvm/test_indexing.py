"""Basic and advanced indexing, Join and Reshape in the LLVM backend."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.link.llvm.linker import LLVMLinker


pytest.importorskip("pytensor_llvm")

# numba_fallback=False: these ops must be native
LLVM = Mode(linker=LLVMLinker(numba_fallback=False), optimizer="fast_run")
REF = Mode(linker="py", optimizer="fast_run")
rng = np.random.default_rng(7)


def compare(inputs, outputs, values, mode=LLVM):
    f = pytensor.function(inputs, outputs, mode=mode, on_unused_input="ignore")
    res = f(*values)
    ref = pytensor.function(inputs, outputs, mode=REF, on_unused_input="ignore")(*values)
    if not isinstance(res, list):
        res, ref = [res], [ref]
    for r, e in zip(res, ref, strict=True):
        assert r.dtype == e.dtype and r.shape == e.shape, (r.shape, e.shape)
        np.testing.assert_allclose(r, e, rtol=1e-13)
    return f


def schedule(f):
    return f.maker.linker.compiled.info()["schedule"]


# ------------------------------------------------------------------ basic indexing


@pytest.mark.parametrize(
    "index",
    [
        lambda x, a, b, c: x[a],
        lambda x, a, b, c: x[-1],
        lambda x, a, b, c: x[1:],
        lambda x, a, b, c: x[:, a],
        lambda x, a, b, c: x[a:b],
        lambda x, a, b, c: x[::-1],
        lambda x, a, b, c: x[b:a:-1],
        lambda x, a, b, c: x[::c, ::-c],
        lambda x, a, b, c: x[-100:100],
        lambda x, a, b, c: x[b:a],  # empty
        lambda x, a, b, c: x[a, b:],
        lambda x, a, b, c: x[:, :, a],
    ],
)
def test_subtensor_patterns(index):
    x = pt.tensor3("x")
    a, b, c = pt.lscalars("a", "b", "c")
    f = compare([x, a, b, c], index(x, a, b, c) * 1.0 + 0.0, [rng.normal(size=(6, 5, 4)), 1, 4, 2])
    assert any("subtensor" in d for d in schedule(f))


def test_subtensor_is_a_view_and_returns_correct_copy():
    x = pt.matrix("x")
    i = pt.lscalar("i")
    f = pytensor.function([x, i], [x[i], x[1:, i]], mode=LLVM)
    v = rng.normal(size=(4, 5))
    a, b = f(v, 2)
    np.testing.assert_array_equal(a, v[2])
    np.testing.assert_array_equal(b, v[1:, 2])
    assert not np.shares_memory(a, v)
    assert any(s.startswith("view of") for _, s, _ in f.maker.linker.compiled.plan())


def test_subtensor_out_of_bounds():
    x = pt.vector("x")
    i = pt.lscalar("i")
    f = pytensor.function([x, i], x[i] * 2, mode=LLVM)
    assert f(np.arange(3.0), -3) == 0.0
    with pytest.raises(Exception, match="out of bounds"):
        f(np.arange(3.0), 3)


@pytest.mark.parametrize("set_", [False, True])
def test_inc_set_subtensor_with_broadcasting(set_):
    x = pt.matrix("x")
    y = pt.vector("y")
    s = pt.dscalar("s")
    i = pt.lscalar("i")
    fn = pt.set_subtensor if set_ else pt.inc_subtensor
    outs = [fn(x[1:, i], y), fn(x[::2], s), fn(x[i], y[:1] * 3), fn(x[:, ::-1], s)]
    compare([x, y, s, i], outs, [rng.normal(size=(5, 4)), rng.normal(size=4), np.array(2.5), 2])


def test_inc_subtensor_casts_y_and_runs_in_place():
    x = pt.matrix("x")
    i = pt.lscalar("i")
    t = pt.exp(x)  # intermediate that dies at the increment
    out = pt.inc_subtensor(t[i], np.float32(1.0)) * 2  # y is float32
    f = compare([x, i], out, [rng.normal(size=(3, 4)), 1])
    assert any("in-place" in s or "slot" in s for _, s, _ in f.maker.linker.compiled.plan())
    assert f.maker.linker.compiled.info()["n_inplace"] >= 1


# ------------------------------------------------------------------ advanced indexing (take)


@pytest.mark.parametrize("idx_dtype", ["int64", "int32"])
def test_take_patterns(idx_dtype):
    x = pt.tensor3("x")
    v = pt.vector("v", dtype=idx_dtype)
    m = pt.matrix("m", dtype=idx_dtype)
    outs = [x[v], x[:, v], x[:, :, v], x[m], x[:, m]]
    vals = [rng.normal(size=(5, 4, 3)), np.array([0, 2, -1, 2], dtype=idx_dtype), np.array([[0, 1], [2, 2]], dtype=idx_dtype)]
    f = compare([x, v, m], outs, vals)
    assert sum("take" in d for d in schedule(f)) == len(outs)


def test_take_out_of_bounds():
    x = pt.vector("x")
    v = pt.lvector("v")
    f = pytensor.function([x, v], x[v] * 1.0, mode=LLVM)
    with pytest.raises(Exception, match="out of bounds"):
        f(np.arange(3.0), np.array([0, 3]))


@pytest.mark.parametrize("set_", [False, True])
def test_inc_take_duplicates_and_broadcasting(set_):
    x = pt.matrix("x")
    v = pt.lvector("v")  # rows
    w = pt.lvector("w")  # columns
    y = pt.matrix("y")
    r = pt.vector("r")
    fn = pt.set_subtensor if set_ else pt.inc_subtensor
    outs = [fn(x[v], y), fn(x[v], r), fn(x[:, w], 1.0)]
    # increments accumulate duplicates (np.add.at); set with duplicates is order dependent
    v_val = np.array([1, 3, 0, 2]) if set_ else np.array([1, 3, 1, -4])
    w_val = np.array([2, 0]) if set_ else np.array([2, -1, 0])
    vals = [rng.normal(size=(4, 3)), v_val, w_val, rng.normal(size=(4, 3)), rng.normal(size=3)]
    compare([x, v, w, y, r], outs, vals)


def test_hierarchical_model_logp_and_grad():
    """Group effects indexed by observation: take forward, inc_take in the gradient."""
    n_groups, n_obs = 8, 200
    mu = pt.dscalar("mu")
    log_tau = pt.dscalar("log_tau")
    theta = pt.vector("theta")
    group = pt.lvector("group")
    y = pt.vector("y")
    tau = pt.exp(log_tau)
    eta = mu + tau * theta
    logp = (-0.5 * (y - eta[group]) ** 2).sum() + (-0.5 * theta**2).sum() - 0.5 * mu**2 + log_tau
    grads = pytensor.grad(logp, [mu, log_tau, theta])
    vals = [np.array(0.2), np.array(-0.3), rng.normal(size=n_groups), rng.integers(0, n_groups, n_obs), rng.normal(size=n_obs)]
    f = compare([mu, log_tau, theta, group, y], [logp, *grads], vals)
    sched = schedule(f)
    assert any("take" in d for d in sched) and any("inc_take" in d for d in sched)


# ------------------------------------------------------------------ join / reshape


def test_join():
    x = pt.matrix("x")
    y = pt.matrix("y")
    z = pt.matrix("z", dtype="float32")
    outs = [pt.concatenate([x, y], axis=0), pt.concatenate([x, y.T, x], axis=1), pt.concatenate([x, z], axis=0)]
    vals = [rng.normal(size=(2, 3)), rng.normal(size=(3, 3)).T.copy()[:2].T.copy(), rng.normal(size=(1, 3)).astype("float32")]
    vals[1] = rng.normal(size=(2, 3))
    compare([x, y, z], outs[:1] + outs[2:], vals)
    compare([x, y], outs[1], [rng.normal(size=(2, 3)), rng.normal(size=(3, 2))])


def test_join_shape_error():
    x = pt.matrix("x")
    y = pt.matrix("y")
    f = pytensor.function([x, y], pt.concatenate([x, y], axis=0), mode=LLVM)
    with pytest.raises(Exception, match="same shape"):
        f(np.zeros((2, 3)), np.zeros((2, 4)))


def test_reshape_view_or_copy():
    x = pt.matrix("x")
    n = pt.lscalar("n")
    t = pt.sort(x, axis=1)  # C intermediate (reshape can't be lifted above Sort) -> view
    outs = [x.reshape((-1,)), t.reshape((n, -1)) + 1, x.T.reshape((-1,))]
    f = compare([x, n], outs, [rng.normal(size=(4, 6)), 3])
    sched = schedule(f)
    assert "view reshape" in sched and "native reshape (allocates)" in sched
    with pytest.raises(Exception, match="cannot reshape"):
        f(rng.normal(size=(4, 6)), 5)


def test_multi_index_to_flat_rewrite():
    # x[i, j] on static axes becomes a flat single-array gather/scatter, which the fused kernel absorbs
    t3 = pt.tensor("t3", shape=(12, 8, 5))
    i, j, y = pt.ivector("i"), pt.ivector("j"), pt.dmatrix("y")
    from pytensor.graph.rewriting.db import RewriteDatabaseQuery
    from pytensor.link.llvm.linker import LLVM_REWRITES

    mode = Mode(LLVMLinker(numba_fallback=False), RewriteDatabaseQuery(include=["fast_run", *LLVM_REWRITES]))
    f = pytensor.function([t3, i, j], (t3[i, j] * 2.0).sum(), mode=mode)
    assert any(s.startswith("FusedElemwise") and "reads=1" in s for s in f.maker.linker.compiled.info()["schedule"])
    T, I, J = (
        rng.normal(size=(12, 8, 5)),
        rng.integers(-12, 12, 40).astype("int32"),
        rng.integers(0, 8, 40).astype("int32"),
    )
    np.testing.assert_allclose(f(T, I, J), (T[I, J] * 2.0).sum())
    g = pytensor.function([t3, i, j, y], pt.inc_subtensor(t3[i, j], y), mode=mode)
    Y, expected = rng.normal(size=(40, 5)), T.copy()
    np.add.at(expected, (I, J), Y)
    np.testing.assert_allclose(g(T, I, J, Y), expected)
    with pytest.raises(IndexError):
        f(T, I, (J + 8).astype("int32"))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_column_take_and_scatter(dtype):
    # last-axis gather/scatter on a 2-d array: native fast path (negatives, duplicates, broadcast rows, bounds)
    x = pt.matrix("x", dtype=dtype)
    idx = pt.lvector("idx")
    yfull, yrow = (pt.matrix("y", dtype=dtype), pt.matrix("yr", dtype=dtype, shape=(1, None)))
    X = rng.normal(size=(7, 9)).astype(dtype)
    I = np.array([0, -1, 3, 3, 8, -9])
    Y, YR = rng.normal(size=(7, 6)).astype(dtype), rng.normal(size=(1, 6)).astype(dtype)
    rtol = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(pytensor.function([x, idx], x[:, idx], mode=LLVM)(X, I), X[:, I], rtol=rtol)
    for y, yv in [(yfull, Y), (yrow, YR)]:
        expected = X.copy()
        np.add.at(expected, (slice(None), I), np.broadcast_to(yv, (7, 6)))
        f = pytensor.function([x, idx, y], pt.inc_subtensor(x[:, idx], y), mode=LLVM)
        np.testing.assert_allclose(f(X, I, yv), expected, rtol=rtol)
    with pytest.raises(IndexError):
        pytensor.function([x, idx], x[:, idx], mode=LLVM)(X, np.array([9]))


def test_bool_mask_indexing_is_native():
    # boolean masks become integer indices (bool_idx_to_nonzero) instead of numba fallbacks
    from pytensor.graph.rewriting.db import RewriteDatabaseQuery
    from pytensor.link.llvm.linker import LLVM_REWRITES

    mode = Mode(LLVMLinker(numba_fallback=False), RewriteDatabaseQuery(include=["fast_run", *LLVM_REWRITES]))
    x, y = pt.dvector("x"), pt.dvector("y")
    mask = np.array([True, False, True, True, False])
    f = pytensor.function([x, y], [x[mask], pt.set_subtensor(x[mask], y)], mode=mode)
    X, Y = np.arange(5.0), np.array([-1.0, -2.0, -3.0])
    taken, written = f(X, Y)
    np.testing.assert_array_equal(taken, X[mask])
    expected = X.copy()
    expected[mask] = Y
    np.testing.assert_array_equal(written, expected)


def test_constant_multi_index_is_static():
    # constant index arrays on static dims: flat index and bounds check resolved at compile time
    x, y = pt.tensor("x", shape=(6, 4, 3)), pt.dmatrix("y")
    I, J = np.array([[0], [-1], [2]]), np.array([1, -4, 3])
    X, Y = rng.normal(size=(6, 4, 3)), rng.normal(size=(3, 3))
    f = compare([x, y], [x[I, J] * 2, pt.inc_subtensor(x[I[:, 0], J], y)], [X, Y])
    assert not [s for s in schedule(f) if "lt" in s or "Reduce{And" in s]
    g = pytensor.function([x], x[np.array([0, 6]), np.array([0, 1])], mode=LLVM)
    with pytest.raises(Exception, match="out of bounds"):
        g(X)


@pytest.mark.parametrize("set_", [False, True])
def test_row_scatter_into_3d_target(set_):
    # N-d targets with contiguous trailing dims take the row scatter path; strided y falls back
    x, y, i = (pt.tensor("x", shape=(3, 2, 5)), pt.tensor("y", shape=(None, 2, 5)), pt.lvector("i"))
    upd = (pt.set_subtensor if set_ else pt.inc_subtensor)(x[i], y)
    I = rng.integers(-3, 3, 40)
    if set_:
        I = np.array([0, -1, 1])
    X, Y = rng.normal(size=(3, 2, 5)), rng.normal(size=(len(I), 2, 10))
    compare([x, y, i], upd, [X, Y[:, :, :5].copy(), I])
    compare([x, y, i], upd, [X, Y[:, :, ::2], I])
