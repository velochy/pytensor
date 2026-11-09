"""Translate a rewritten FunctionGraph into the native graph builder of `pytensor_llvm`.

Each Op gets a converter registered with ``llvm_build.register(OpType)``. A converter
receives the node and the builder ids of its inputs and outputs and adds the matching
node(s) to the builder, which validates them immediately. All decisions about buffers,
layouts, in-place updates and kernels are made natively by the planner.
"""

from __future__ import annotations

from functools import singledispatch

import numpy as np

import pytensor.scalar.basic as ps
import pytensor.tensor as _pt
import pytensor.tensor.basic as _ptb
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.graph.basic import Constant, Variable
from pytensor.raise_op import CheckAndRaise
from pytensor.scan.op import Scan
from pytensor.scalar.basic import Composite, ScalarOp, ScalarType
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.basic import Join
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.sort import SortOp
from pytensor.tensor.rewriting.subtensor import flat_multi_index
from pytensor.tensor.type import TensorType


SUPPORTED_DTYPES = frozenset(
    ["float64", "float32", "int64", "int32", "int16", "int8", "uint8", "bool"]
)


def _native():
    try:
        import pytensor_llvm
    except ImportError as e:
        raise ImportError(
            "The LLVM backend needs the `pytensor_llvm` extension "
            "(build it with rust/pytensor-llvm/build.sh)."
        ) from e
    return pytensor_llvm


class GraphConversion:
    """Maps PyTensor variables to builder ids while a graph is being converted."""

    def __init__(self, numba_fallback=True):
        self.native = _native()
        self.builder = self.native.GraphBuilder()
        self.ids: dict[Variable, int] = {}
        self.consts: dict[tuple, tuple[int, np.ndarray]] = {}
        self.numba_fallback = numba_fallback

    def convert_node(self, node):
        """Add `node`, falling back to its numba implementation if needed."""
        ins = [self.var(i) for i in node.inputs]
        outs = [self.var(o) for o in node.outputs]
        try:
            llvm_build(node.op, node, self, ins, outs)
        except NotImplementedError:
            if not self.numba_fallback:
                raise
            from pytensor.link.llvm.numba_fallback import numba_fallback_kernel

            kernel = numba_fallback_kernel(node)
            self.builder.foreign(kernel.address, ins, outs, str(node.op), kernel)

    def inline(self, inner, inputs, outputs):
        """Convert the nodes of an inner graph in place of an OpFromGraph node.

        The inner variables get their own id scope: the same inner graph can be
        shared by several nodes (e.g. after merging), and must not see the ids
        assigned while inlining another of them.
        """
        outer_ids, self.ids = self.ids, {}
        try:
            self._inline(inner, inputs, outputs)
        finally:
            self.ids = outer_ids

    def _inline(self, inner, inputs, outputs):
        for iv, i in zip(inner.inputs, inputs, strict=True):
            self.ids[iv] = i
        copies = []
        for ov, o in zip(inner.outputs, outputs, strict=True):
            if ov.owner is None or ov in self.ids:
                copies.append((ov, o))  # inner input, constant or repeated output
            else:
                self.ids[ov] = o  # the producing inner node writes the outer output
        for n in inner.toposort():
            self.convert_node(n)
        for ov, o in copies:
            self.builder.copy(self.var(ov), o)

    def var(self, v: Variable) -> int:
        if v in self.ids:
            return self.ids[v]
        t = v.type
        if isinstance(t, NoneTypeT) and isinstance(v, Constant):
            return None  # e.g. unknown dims of SpecifyShape; converters must not use it
        if isinstance(t, ScalarType):
            # PyTensor scalars are 0-d tensors here
            bcast, shape = [], []
        elif isinstance(t, TensorType):
            bcast = [bool(b) for b in t.broadcastable]
            shape = [None if s is None else int(s) for s in t.shape]
        else:
            raise NotImplementedError(f"LLVM backend: variables of type {t} are not supported")
        if t.dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(f"LLVM backend: dtype {t.dtype} is not supported")
        if isinstance(v, Constant):
            i = self.constant(np.asarray(v.data, dtype=t.dtype), bcast, shape)
        else:
            i = self.builder.variable(t.dtype, bcast, shape)
        self.ids[v] = i
        return i

    def constant(self, data: np.ndarray, bcast, shape) -> int:
        """Builder id of a constant; buffers are shared, and permuted contiguous data stays a view."""
        perm = None
        if data.ndim > 1 and not data.flags.c_contiguous and min(data.strides) > 0:
            p = tuple(int(i) for i in np.argsort([-s for s in data.strides], kind="stable"))
            if data.transpose(p).flags.c_contiguous:
                data, perm = data.transpose(p), p
        key = (data.__array_interface__["data"][0], data.shape, data.strides, data.dtype.str)
        if key not in self.consts:  # holding `data` keeps its address from being reused
            self.consts[key] = (self.builder.constant(data, [bcast[i] for i in perm] if perm else bcast), data)
        i = self.consts[key][0]
        if perm is None:
            return i
        out = self.builder.variable(data.dtype.name, bcast, shape)
        self.builder.dimshuffle(np.argsort(perm).tolist(), i, out)
        return out

    def cast(self, v: Variable, dtype: str) -> int:
        """Builder id of `v` converted to `dtype` (adds a cast Elemwise if needed)."""
        i = self.var(v)
        if v.type.dtype == dtype:
            return i
        if dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(f"LLVM backend: dtype {dtype} is not supported")
        t = v.type
        bcast = [bool(b) for b in t.broadcastable] if isinstance(t, TensorType) else []
        shape = [None if s is None else int(s) for s in t.shape] if isinstance(t, TensorType) else []
        out = self.builder.variable(dtype, bcast, shape)
        sg = self.native.ScalarGraph([t.dtype])
        sg.set_outputs([sg.apply("cast", [sg.input(0)], dtype)])
        self.builder.elemwise(sg, [i], [out])
        return out


# ------------------------------------------------------------------ scalar graphs

_SCALAR_NAME = {"scalarmaximum": "maximum", "scalarminimum": "minimum"}


def _scalar_op_name(op) -> str:
    name = type(op).__name__.lower()
    return _SCALAR_NAME.get(name, name)


def scalar_graph(native, op, input_types):
    """Build a native ScalarGraph for a scalar op (Composite or single op)."""
    sg = native.ScalarGraph([t.dtype for t in input_types])
    inputs = [sg.input(i) for i in range(len(input_types))]
    if isinstance(op, Composite):
        outs = _emit_composite(sg, op, inputs)
    else:
        node = op.make_node(*[t() for t in input_types])
        outs = [_emit_scalar_node(sg, node, inputs)]
    sg.set_outputs(outs)
    return sg


def _scalar_ref(sg, v, refs):
    if v in refs:
        return refs[v]
    if isinstance(v, Constant):
        if v.type.dtype not in SUPPORTED_DTYPES:
            raise NotImplementedError(f"LLVM backend: scalar constant of dtype {v.type.dtype}")
        return sg.constant(float(np.asarray(v.data)), v.type.dtype)
    raise NotImplementedError(f"LLVM backend: unresolved scalar variable {v}")


def _emit_composite(sg, comp: Composite, inputs):
    refs = dict(zip(comp.fgraph.inputs, inputs, strict=True))
    for node in comp.fgraph.toposort():
        args = [_scalar_ref(sg, i, refs) for i in node.inputs]
        if isinstance(node.op, Composite):  # nested composites are inlined
            outs = _emit_composite(sg, node.op, args)
        else:
            outs = [_emit_scalar_node(sg, node, args)]
        refs.update(zip(node.outputs, outs, strict=True))
    return [_scalar_ref(sg, o, refs) for o in comp.fgraph.outputs]


def _emit_scalar_node(sg, node, args):
    if len(node.outputs) != 1:
        raise NotImplementedError(f"LLVM backend: multi-output scalar op {node.op}")
    dtype = node.outputs[0].type.dtype
    if dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"LLVM backend: scalar op {node.op} with dtype {dtype}")
    return sg.apply(_scalar_op_name(node.op), args, dtype)


# ------------------------------------------------------------------ op converters


@singledispatch
def llvm_build(op, node, conv: GraphConversion, inputs: list[int], outputs: list[int]):
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


@llvm_build.register(Elemwise)
def _(op, node, conv, inputs, outputs):
    types = [ps.get_scalar_type(i.type.dtype) for i in node.inputs]
    sg = scalar_graph(conv.native, op.scalar_op, types)
    conv.builder.elemwise(sg, inputs, outputs)


@llvm_build.register(ScalarOp)
def _(op, node, conv, inputs, outputs):
    # a bare scalar op is a 0-d Elemwise
    sg = scalar_graph(conv.native, op, [i.type for i in node.inputs])
    conv.builder.elemwise(sg, inputs, outputs)


_REDUCE_NAME = {
    ps.Add: "add",
    ps.Mul: "mul",
    ps.Maximum: "max",
    ps.Minimum: "min",
    ps.AND: "and",
    ps.OR: "or",
}


@llvm_build.register(CAReduce)
def _(op, node, conv, inputs, outputs):
    name = _REDUCE_NAME.get(type(op.scalar_op))
    if name is None:
        raise NotImplementedError(f"LLVM backend: CAReduce with {op.scalar_op}")
    ndim = node.inputs[0].type.ndim
    axis = range(ndim) if op.axis is None else [a % ndim for a in op.axis]
    conv.builder.reduce(name, sorted(axis), inputs[0], outputs[0])


@llvm_build.register(DimShuffle)
def _(op, node, conv, inputs, outputs):
    order = [None if o == "x" else int(o) for o in op.new_order]
    conv.builder.dimshuffle(order, inputs[0], outputs[0])


@llvm_build.register(ViewOp)
@llvm_build.register(SpecifyShape)
def _(op, node, conv, inputs, outputs):
    # identity view; SpecifyShape's runtime shape check is not implemented yet
    conv.builder.dimshuffle(list(range(node.inputs[0].type.ndim)), inputs[0], outputs[0])


@llvm_build.register(ScalarFromTensor)
@llvm_build.register(TensorFromScalar)
def _(op, node, conv, inputs, outputs):
    conv.builder.dimshuffle([], inputs[0], outputs[0])


@llvm_build.register(Dot)
def _(op, node, conv, inputs, outputs):
    # the native matmul is float-only and same-dtype: upcast mixed operands like numpy does
    dtype = node.outputs[0].type.dtype
    if dtype not in ("float64", "float32"):
        raise NotImplementedError(f"LLVM backend: Dot with {dtype} output")
    a, b = (conv.cast(v, dtype) for v in node.inputs)
    conv.builder.dot(a, b, outputs[0])


@llvm_build.register(SortOp)
def _(op, node, conv, inputs, outputs):
    axis = int(op.axis) % node.inputs[0].type.ndim
    conv.builder.sort(inputs[0], outputs[0], axis, op.kind in ("stable", "mergesort"))


from pytensor.tensor.linalg.decomposition.cholesky import Cholesky  # noqa: E402
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular  # noqa: E402


@llvm_build.register(Cholesky)
def _(op, node, conv, inputs, outputs):
    out_dtype = node.outputs[0].type.dtype
    conv.builder.cholesky(conv.cast(node.inputs[0], out_dtype), outputs[0], bool(op.lower))


@llvm_build.register(SolveTriangular)
def _(op, node, conv, inputs, outputs):
    out_dtype = node.outputs[0].type.dtype
    a, b = (conv.cast(v, out_dtype) for v in node.inputs)
    conv.builder.solve_triangular(a, b, outputs[0], bool(op.lower), bool(op.unit_diagonal))


@llvm_build.register(DeepCopyOp)
def _(op, node, conv, inputs, outputs):
    conv.builder.copy(inputs[0], outputs[0])


@llvm_build.register(Shape_i)
def _(op, node, conv, inputs, outputs):
    conv.builder.shape_i(inputs[0], outputs[0], int(op.i))


@llvm_build.register(Shape)
def _(op, node, conv, inputs, outputs):
    conv.builder.shape(inputs[0], outputs[0])


@llvm_build.register(MakeVector)
def _(op, node, conv, inputs, outputs):
    conv.builder.make_vector(inputs, outputs[0])


@llvm_build.register(CheckAndRaise)
def _(op, node, conv, inputs, outputs):
    msg = str(getattr(op, "msg", "Assertion failed"))
    conv.builder.check_and_raise(inputs[0], inputs[1:], outputs[0], msg)


def _idx_list(idx_list):
    """PyTensor's idx_list in the builder's encoding."""
    return [e if isinstance(e, int) else (e.start, e.stop, e.step) for e in idx_list]


def _take_axis(idx_list, index_vars):
    """Axis if the index is a single integer array on one axis with all others whole."""
    ints = [d for d, e in enumerate(idx_list) if isinstance(e, int)]
    if len(ints) != 1 or len(index_vars) != 1:
        return None
    if any(e != slice(None) for d, e in enumerate(idx_list) if d != ints[0]):
        return None
    t = index_vars[0].type
    if not isinstance(t, TensorType) or t.dtype == "bool" or not t.dtype.startswith(("int", "uint")):
        return None
    return ints[0]


@llvm_build.register(Subtensor)
def _(op, node, conv, inputs, outputs):
    conv.builder.subtensor(inputs[0], _idx_list(op.idx_list), inputs[1:], outputs[0])


@llvm_build.register(IncSubtensor)
def _(op, node, conv, inputs, outputs):
    x, y, *idx = node.inputs
    y_id = conv.cast(y, x.type.dtype)
    conv.builder.inc_subtensor(
        inputs[0], y_id, _idx_list(op.idx_list), inputs[2:], outputs[0], op.set_instead_of_inc
    )


@llvm_build.register(AdvancedSubtensor)
def _(op, node, conv, inputs, outputs):
    axis = _take_axis(op.idx_list, node.inputs[1:])
    if axis is None:
        multi = _multi_index(op, node.inputs[1:])
        if multi is None:
            raise NotImplementedError(f"LLVM backend: advanced indexing pattern {op}")
        return _lower_adv_read(conv, node, inputs, outputs, *multi)
    conv.builder.take(inputs[0], inputs[1], axis, outputs[0])


@llvm_build.register(AdvancedIncSubtensor)
def _(op, node, conv, inputs, outputs):
    axis = _take_axis(op.idx_list, node.inputs[2:])
    if op.ignore_duplicates and not op.set_instead_of_inc:
        # ignore_duplicates increments follow `x[idx] += y` (no accumulation)
        raise NotImplementedError(f"LLVM backend: advanced indexing pattern {op}")
    if axis is None:
        multi = _multi_index(op, node.inputs[2:])
        if multi is None:
            raise NotImplementedError(f"LLVM backend: advanced indexing pattern {op}")
        return _lower_adv_inc(conv, node, inputs, outputs, *multi, op.set_instead_of_inc)
    x, y, _ = node.inputs
    y_id = conv.cast(y, x.type.dtype)
    conv.builder.inc_take(inputs[0], y_id, inputs[2], axis, outputs[0], op.set_instead_of_inc)


@llvm_build.register(Join)
def _(op, node, conv, inputs, outputs):
    dtype = node.outputs[0].type.dtype
    ids = [conv.cast(v, dtype) for v in node.inputs]
    conv.builder.join(ids, int(op.axis), outputs[0])


@llvm_build.register(Reshape)
def _(op, node, conv, inputs, outputs):
    conv.builder.reshape(inputs[0], inputs[1], outputs[0])


def _fused_spec(op, node):
    """Translate a FusedElemwise into the native builder's description, or raise
    NotImplementedError for patterns the native fused kernel doesn't cover."""
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise  # noqa: F401

    inner = op.fgraph
    [E] = [n for n in inner.apply_nodes if isinstance(n.op, Elemwise)]
    ndim = E.outputs[0].type.ndim
    n_idx = len(op.indexed_inputs)
    if len(op.indexed_outputs) not in (0, n_idx):
        raise NotImplementedError("FusedElemwise: inconsistent index specs")
    # one target buffer per written output (an index can write several),
    # ordered by output position
    written = sorted(int(s) for e in op.indexed_outputs if e is not None for s in e[0])
    n_elem = len(node.inputs) - n_idx - len(written)
    if len(E.outputs) != len(node.outputs):
        raise NotImplementedError("FusedElemwise: unexpected inner structure")

    def check_index(p):
        t = node.inputs[p].type
        if not isinstance(t, TensorType) or t.ndim != 1 or t.dtype == "bool" or not t.dtype.startswith(("int", "uint")):
            raise NotImplementedError("FusedElemwise: only 1-d integer index arrays are fused")

    # Elemwise inputs may share one source array read through several indices, so the kernel gets one
    # operand per Elemwise input (sources repeated); `src_pos` maps them onto the node's inputs.
    n_e = len(E.inputs)
    reads = [None] * n_e
    for k, entry in enumerate(op.indexed_inputs):
        if entry is None:
            continue
        sources, axis = entry
        check_index(n_elem + k)
        for s in sources:
            if reads[s] is not None:
                # x[i, j]: one source read through several index arrays
                raise NotImplementedError("FusedElemwise: multi-index reads are not fused natively")
            reads[s] = (n_e + k, int(axis))
    src_pos = []
    for j, e_in in enumerate(E.inputs):
        o = e_in.owner
        if reads[j] is not None:
            if not (o and isinstance(o.op, AdvancedSubtensor) and o.inputs[0] in inner.inputs):
                raise NotImplementedError("FusedElemwise: unexpected indexed read")
            src_pos.append(inner.inputs.index(o.inputs[0]))
        else:
            # scalars enter the loop through TensorFromScalar / ExpandDims chains
            root = e_in
            while root not in inner.inputs and root.owner and len(root.owner.inputs) == 1:
                root = root.owner.inputs[0]
            if root not in inner.inputs or (
                root is not e_in
                and not (
                    isinstance(node.inputs[inner.inputs.index(root)].type, ScalarType) and all(e_in.type.broadcastable)
                )
            ):
                raise NotImplementedError("FusedElemwise: transformed Elemwise input")
            src_pos.append(inner.inputs.index(root))

    outs = [("plain",)] * len(node.outputs)
    for i, spec in enumerate(op.reduced_outputs):
        if spec is None:
            continue
        scalar_op, axes, _identity, acc_dtype = spec
        name = _REDUCE_NAME.get(type(scalar_op))
        if name is None or not (acc_dtype == node.outputs[i].type.dtype == E.outputs[i].type.dtype):
            raise NotImplementedError("FusedElemwise: reduction not supported natively")
        outs[i] = ("reduce", name, [int(a) for a in axes])
    for k, entry in enumerate(op.indexed_outputs):
        if entry is None:
            continue
        sources, axis, mode = entry
        check_index(n_elem + k)
        for s in sources:
            if outs[s][0] != "plain":
                raise NotImplementedError("FusedElemwise: multi-index writes are not fused natively")
            target = n_e + n_idx + written.index(int(s))
            outs[s] = ("write", n_e + k, int(axis), target, mode == "set")
    for i, o in enumerate(node.outputs):
        if o.type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError("FusedElemwise: output dtype differs from the loop's")
    operands = [*src_pos, *range(n_elem, len(node.inputs))]
    return E, ndim, n_e, reads, outs, operands


def _register_fused():
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    @llvm_build.register(FusedElemwise)
    def _(op, node, conv, inputs, outputs):
        try:
            E, ndim, n_e, reads, outs, operands = _fused_spec(op, node)
        except NotImplementedError:
            # still native, just not fused
            return conv.inline(op.fgraph, inputs, outputs)
        types = [ps.get_scalar_type(i.type.dtype) for i in E.inputs]
        sg = scalar_graph(conv.native, E.op.scalar_op, types)
        conv.builder.fused_elemwise(sg, [inputs[p] for p in operands], outputs, ndim, n_e, reads, outs)


_register_fused()


def _static_untraced_step(op, node):
    """Step graph with untraced states typed by their static outer shape, if every step keeps it."""
    ins, outs, nodes = op.fgraph.inputs, op.fgraph.outputs, op.fgraph.toposort()
    memo = {
        i: i.type.clone(shape=o.type.shape)(name=i.name)
        for i, o in zip(op.inner_untraced_sit_sot(ins), op.outer_untraced_sit_sot(node.inputs), strict=True)
        if isinstance(i.type, TensorType) and i.type.shape != o.type.shape and None not in o.type.shape
    }
    if not memo:
        return ins, nodes, outs
    new_nodes = []
    try:
        for n in nodes:
            new_in = [memo.get(i, i) for i in n.inputs]
            nn = n.op.make_node(*new_in) if any(a is not b for a, b in zip(new_in, n.inputs)) else n
            memo.update(zip(n.outputs, nn.outputs, strict=True))
            new_nodes.append(nn)
    except Exception:
        return ins, nodes, outs
    new_ins, new_outs = [memo.get(i, i) for i in ins], [memo.get(o, o) for o in outs]
    kept = all(
        a.type.shape == b.type.shape
        for a, b in zip(op.inner_untraced_sit_sot(new_ins), op.inner_untraced_sit_sot_outs(new_outs), strict=True)
    )
    return (new_ins, new_nodes, new_outs) if kept else (ins, nodes, outs)


@llvm_build.register(Scan)
def _(op, node, conv, inputs, outputs):
    # the step function becomes its own graph (compiled natively; see scan.rs)
    inner = GraphConversion(numba_fallback=conv.numba_fallback)
    ins, nodes, outs = _static_untraced_step(op, node)
    for i in ins:
        inner.builder.add_input(inner.var(i))
    for n in nodes:
        inner.convert_node(n)
    for o in outs:
        inner.builder.add_output(inner.var(o))
    info = op.info
    taps = lambda slices: [[int(t) for t in s] for s in slices]  # noqa: E731
    conv.builder.scan(
        inner.builder,
        inputs,
        outputs,
        n_seqs=info.n_seqs,
        mit_mot_in=taps(info.mit_mot_in_slices),
        mit_mot_out=taps(info.mit_mot_out_slices),
        mit_sot_in=taps(info.mit_sot_in_slices),
        sit_sot_in=taps(info.sit_sot_in_slices),
        n_untraced=info.n_untraced_sit_sot,
        n_nit_sot=info.n_nit_sot,
        n_non_seqs=info.n_non_seqs,
        as_while=info.as_while,
    )


@llvm_build.register(OpFromGraph)
def _(op, node, conv, inputs, outputs):
    # FusedElemwise, ScalarizedElemwise, ScalarCAReduce, user OpFromGraphs, ...
    conv.inline(op.fgraph, inputs, outputs)


@llvm_build.register(ARange)
def _(op, node, conv, inputs, outputs):
    conv.builder.arange(*inputs, outputs[0])


@llvm_build.register(AllocEmpty)
def _(op, node, conv, inputs, outputs):
    conv.builder.alloc_empty(inputs, outputs[0])


@llvm_build.register(Alloc)
def _(op, node, conv, inputs, outputs):
    conv.builder.alloc(inputs[0], inputs[1:], outputs[0])


# ------------------------------------------------------------------ whole graphs


def _written_input(node):
    """Position of the input `node` updates (its output may reuse that buffer), else None."""
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    if isinstance(node.op, IncSubtensor | AdvancedIncSubtensor):
        return 0
    if isinstance(node.op, FusedElemwise):
        n_written = sum(len(e[0]) for e in node.op.indexed_outputs if e is not None)
        return len(node.inputs) - n_written if n_written == 1 else None
    return None


def _writer_last_order(fgraph):
    """Toposort that runs the other readers of a buffer before the node that updates it, where the
    dependencies allow: the planner writes in place only into a buffer that dies at the writer, as
    PyTensor's DestroyHandler orders destroyers after other clients for the C backend."""
    from pytensor.graph.traversal import toposort_with_orderings

    order = fgraph.toposort()
    pos = {n: k for k, n in enumerate(order)}

    def depends_on(c, w):  # is w an ancestor of c? (nodes before w in the toposort can't be)
        stack, seen = [c], set()
        while stack:
            n = stack.pop()
            if n is w:
                return True
            for i in n.inputs:
                o = i.owner
                if o is not None and o not in seen and pos[o] >= pos[w]:
                    seen.add(o)
                    stack.append(o)
        return False

    orderings: dict = {}
    writers: dict = {}
    for n in order:
        k = _written_input(n)
        if k is not None and n.inputs[k].owner is not None:
            writers[n.inputs[k]] = n  # the last writer of each buffer gets it
    for x, w in writers.items():
        before = [c for c, _ in fgraph.clients[x] if not isinstance(c, str) and c is not w and not depends_on(c, w)]
        if before:
            orderings[w] = before
    if not orderings:
        return order
    try:  # orderings that are each acyclic can still form a cycle together
        new = list(toposort_with_orderings(fgraph.outputs, blockers=fgraph.inputs, orderings=orderings))
    except ValueError:
        return order
    return new if len(new) == len(order) and set(new) == set(order) else order


def fgraph_to_llvm(
    fgraph,
    *,
    fastmath=True,
    opt_level=3,
    cache=True,
    dump_ir=None,
    segments=True,
    segment_max_elems=64,
    numba_fallback=True,
):
    """Convert and compile `fgraph`; returns a native ``CompiledFunction``.

    Ops without an LLVM implementation run through their numba implementation if
    `numba_fallback` is set (see `numba_fallback.py`).
    """
    conv = GraphConversion(numba_fallback=numba_fallback)
    b = conv.builder
    for i in fgraph.inputs:
        b.add_input(conv.var(i))
    for node in _writer_last_order(fgraph):
        conv.convert_node(node)
    for o in fgraph.outputs:
        b.add_output(conv.var(o))
    return b.compile(
        fastmath=fastmath,
        opt_level=opt_level,
        cache=cache,
        dump_ir=dump_ir,
        segments=segments,
        segment_max_elems=segment_max_elems,
    )


# ---------------------------------------------------------------- lowered ops
#
# Ops expressed through other supported ops: a small PyTensor graph built on the
# node's inputs, converted in place of the node.


def _lower(conv, node, inputs, outputs, build):
    """Convert `build(*node.inputs)` (a list of PyTensor outputs) in place of `node`.

    Constant inputs stay constants in the built graph (so slices with constant bounds
    are static), the others are replaced by fresh variables bound to `inputs`.
    """
    from pytensor.graph.fg import FunctionGraph

    fresh = [v if isinstance(v, Constant) else v.type() for v in node.inputs]
    outs = build(*fresh)
    outs = [o if o.type == n.type else o.astype(n.type.dtype) for o, n in zip(outs, node.outputs, strict=True)]
    # outputs that are inputs or constants are copied by `inline`
    fg_inputs = [f for f, v in zip(fresh, node.inputs, strict=True) if not isinstance(v, Constant)]
    fg_ids = [i for i, v in zip(inputs, node.inputs, strict=True) if not isinstance(v, Constant)]
    fg = FunctionGraph(fg_inputs, outs, clone=False)
    conv.inline(fg, fg_ids, outputs)


@llvm_build.register(_ptb.Split)
def _(op, node, conv, inputs, outputs):
    x, splits = node.inputs
    axis = int(op.axis)

    def build(x, splits):
        if isinstance(splits, Constant):
            sizes = [int(s) for s in np.asarray(splits.data)]
            starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
            bounds = [(int(a), int(a) + s) for a, s in zip(starts, sizes, strict=True)]
        else:
            ends = _pt.cumsum(splits)
            bounds = [(ends[k] - splits[k], ends[k]) for k in range(op.len_splits)]
        lead = (slice(None),) * axis
        return [x[(*lead, slice(a, b))] for a, b in bounds]

    _lower(conv, node, inputs, outputs, build)


def _multi_index(op, index_vars):
    """For x[i0, ..., ik] with integer index arrays on consecutive axes a..a+k (full
    slices elsewhere): (a, index vars), else None."""
    idx = op.idx_list
    ints = [d for d, e in enumerate(idx) if isinstance(e, int)]
    if len(ints) < 2 or ints != list(range(ints[0], ints[0] + len(ints))):
        return None
    if any(e != slice(None) for d, e in enumerate(idx) if d not in ints):
        return None
    vars_ = [index_vars[idx[d]] for d in ints]
    if any(not isinstance(v.type, TensorType) or v.type.dtype == "bool" or not v.type.dtype.startswith(("int", "uint")) for v in vars_):
        return None
    return ints[0], vars_


def _dim(x, d):
    return x.type.shape[d] if x.type.shape[d] is not None else x.shape[d]


def _lower_adv_read(conv, node, inputs, outputs, a, index_vars):
    pos = [node.inputs.index(v) for v in index_vars]

    def build(x, *idx):
        ivars = [idx[p - 1] for p in pos]
        xr, flat = flat_multi_index(x, a, ivars)
        lead = (slice(None),) * a
        taken = xr[(*lead, flat.ravel())]
        if flat.type.ndim == 1:
            return [taken]
        bshape = [_dim(flat, d) for d in range(flat.type.ndim)]
        out_shape = [
            *[_dim(x, d) for d in range(a)],
            *bshape,
            *[_dim(x, d) for d in range(a + len(ivars), x.type.ndim)],
        ]
        return [taken.reshape(out_shape, ndim=node.outputs[0].type.ndim)]

    _lower(conv, node, inputs, outputs, build)


def _lower_adv_inc(conv, node, inputs, outputs, a, index_vars, set_):
    pos = [node.inputs.index(v) for v in index_vars]

    def build(x, y, *idx):
        ivars = [idx[p - 2] for p in pos]
        xr, flat = flat_multi_index(x, a, ivars)
        lead = (slice(None),) * a
        target = xr[(*lead, flat.ravel())]
        # y broadcast to the indexed region, then flattened like it
        region = [
            *[_dim(x, d) for d in range(a)],
            *[_dim(flat, d) for d in range(flat.type.ndim)],
            *[_dim(x, d) for d in range(a + len(ivars), x.type.ndim)],
        ]
        y = _pt.cast(y, x.type.dtype)
        full = y.type.ndim == len(region) and not any(y.type.broadcastable)
        yb = y if full else _pt.alloc(y, *region)
        # 1-d indices leave the region's shape as is: no reshape (a copy, as y's layout is unknown)
        yf = (
            yb
            if flat.type.ndim == 1
            else yb.reshape([_dim(target, d) for d in range(target.type.ndim)], ndim=target.type.ndim)
        )
        upd = _pt.set_subtensor(target, yf) if set_ else _pt.inc_subtensor(target, yf)
        return [upd.reshape([_dim(x, d) for d in range(x.type.ndim)], ndim=x.type.ndim)]

    _lower(conv, node, inputs, outputs, build)


from pytensor.tensor.extra_ops import CumOp  # noqa: E402


@llvm_build.register(CumOp)
def _(op, node, conv, inputs, outputs):
    out_dtype = node.outputs[0].type.dtype
    x = node.inputs[0]
    if x.type.ndim == 0:
        raise NotImplementedError("LLVM backend: CumOp on a 0-d input")
    conv.builder.cumop(conv.cast(x, out_dtype), outputs[0], int(op.axis) % x.type.ndim, op.mode == "mul")


@llvm_build.register(_ptb.ExtractDiag)
def _(op, node, conv, inputs, outputs):
    nd = node.inputs[0].type.ndim
    conv.builder.extract_diag(inputs[0], outputs[0], int(op.offset), int(op.axis1) % nd, int(op.axis2) % nd)
