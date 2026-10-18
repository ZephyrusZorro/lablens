"""Minimal ONNX protobuf writer (no `onnx` package needed).

Implements just the parts of the ONNX schema that a CNN detector uses: ModelProto, GraphProto,
NodeProto, AttributeProto, TensorProto (raw_data), ValueInfoProto and OperatorSetIdProto, encoded
directly in protobuf wire format. Field numbers follow onnx/onnx.proto.
"""
from __future__ import annotations

import struct
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

# TensorProto.DataType
FLOAT, INT64, FLOAT16 = 1, 7, 10
_NP_TO_ONNX = {np.dtype(np.float32): FLOAT, np.dtype(np.int64): INT64, np.dtype(np.float16): FLOAT16}

# AttributeProto.AttributeType
_A_FLOAT, _A_INT, _A_STRING, _A_TENSOR, _A_FLOATS, _A_INTS = 1, 2, 3, 4, 6, 7


# --------------------------------------------------------------------------- wire format
def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64  # two's complement for negative int64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def f_int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(int(v))


def f_bytes(field: int, v: Union[bytes, str]) -> bytes:
    if isinstance(v, str):
        v = v.encode("utf-8")
    return _key(field, 2) + _varint(len(v)) + v


def f_float(field: int, v: float) -> bytes:
    return _key(field, 5) + struct.pack("<f", float(v))


def f_msg(field: int, payload: bytes) -> bytes:
    return f_bytes(field, payload)


# --------------------------------------------------------------------------- messages
def tensor(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    dt = _NP_TO_ONNX[arr.dtype]
    out = b"".join(f_int(1, d) for d in arr.shape)
    out += f_int(2, dt) + f_bytes(8, name) + f_bytes(9, arr.tobytes())
    return out


def attribute(name: str, value) -> bytes:
    out = f_bytes(1, name)
    if isinstance(value, bool) or isinstance(value, (int, np.integer)):
        return out + f_int(3, int(value)) + f_int(20, _A_INT)
    if isinstance(value, float):
        return out + f_float(2, value) + f_int(20, _A_FLOAT)
    if isinstance(value, str):
        return out + f_bytes(4, value) + f_int(20, _A_STRING)
    if isinstance(value, np.ndarray):
        return out + f_msg(5, tensor("", value)) + f_int(20, _A_TENSOR)
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, np.integer)) for v in value):
            return out + b"".join(f_int(8, int(v)) for v in value) + f_int(20, _A_INTS)
        return out + b"".join(f_float(7, float(v)) for v in value) + f_int(20, _A_FLOATS)
    raise TypeError(f"unsupported attribute {name}={value!r}")


def node(op: str, inputs: Sequence[str], outputs: Sequence[str], name: str = "", **attrs) -> bytes:
    out = b"".join(f_bytes(1, i) for i in inputs) + b"".join(f_bytes(2, o) for o in outputs)
    if name:
        out += f_bytes(3, name)
    out += f_bytes(4, op)
    out += b"".join(f_msg(5, attribute(k, v)) for k, v in attrs.items())
    return out


def value_info(name: str, elem_type: int, shape: Sequence[Union[int, str, None]]) -> bytes:
    dims = b""
    for d in shape:
        if isinstance(d, int):
            dims += f_msg(1, f_int(1, d))
        else:
            dims += f_msg(1, f_bytes(2, d or "?"))
    tensor_type = f_int(1, elem_type) + f_msg(2, dims)
    return f_bytes(1, name) + f_msg(2, f_msg(1, tensor_type))


def model(nodes: List[bytes], initializers: List[bytes], inputs: List[bytes], outputs: List[bytes],
          graph_name: str = "graph", opset: int = 17, ir_version: int = 8,
          metadata: Optional[Dict[str, str]] = None, producer: str = "lablens") -> bytes:
    graph = b"".join(f_msg(1, n) for n in nodes) + f_bytes(2, graph_name)
    graph += b"".join(f_msg(5, t) for t in initializers)
    graph += b"".join(f_msg(11, i) for i in inputs) + b"".join(f_msg(12, o) for o in outputs)
    out = f_int(1, ir_version) + f_bytes(2, producer) + f_bytes(3, "1.0") + f_msg(7, graph)
    out += f_msg(8, f_bytes(1, "") + f_int(2, opset))
    for k, v in (metadata or {}).items():
        out += f_msg(14, f_bytes(1, k) + f_bytes(2, v))
    return out


class GraphBuilder:
    """Tiny helper that hands out unique tensor names and collects nodes / initializers."""

    def __init__(self):
        self.nodes: List[bytes] = []
        self.inits: List[bytes] = []
        self._n = 0
        self.const_cache: Dict[bytes, str] = {}

    def uid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def init(self, arr: np.ndarray, prefix: str = "w") -> str:
        name = self.uid(prefix)
        self.inits.append(tensor(name, arr))
        return name

    def const(self, arr: np.ndarray) -> str:
        key = arr.dtype.str.encode() + str(arr.shape).encode() + arr.tobytes()
        if key not in self.const_cache:
            self.const_cache[key] = self.init(arr, "c")
        return self.const_cache[key]

    def op(self, op: str, inputs: Iterable[str], n_out: int = 1, **attrs):
        outs = [self.uid(op.lower()) for _ in range(n_out)]
        self.nodes.append(node(op, list(inputs), outs, **attrs))
        return outs[0] if n_out == 1 else outs
