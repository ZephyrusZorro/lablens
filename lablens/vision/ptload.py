"""Torch-free, code-execution-free loader for PyTorch / Ultralytics `.pt` checkpoints.

Why: `torch.load` unpickles arbitrary Python (a .pt from the internet can run code on your machine),
and it needs the 700 MB torch install. This loader:

* reads the zip container directly,
* rebuilds tensors as numpy arrays straight from the storage blobs (zero-copy views),
* replaces every class referenced by the pickle with an inert `Stub` that only records state,
  so nothing from the file is ever executed. Only a tiny allow-list of pure-data builtins is real.

The result is a tree of `Stub` objects mirroring the nn.Module hierarchy: `obj.state` holds the
module's __dict__ (`_modules`, `_parameters`, `_buffers`, hyper-parameters like `stride`, `f`, `i`).
"""
from __future__ import annotations

import collections
import io
import pickle
import zipfile
from typing import Any, Dict, Tuple

import numpy as np

_STORAGE_DTYPES = {
    "HalfStorage": np.float16, "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "BFloat16Storage": np.uint16,  # converted below
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8,
    "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}
_TORCH_DTYPES = {
    "float16": np.float16, "half": np.float16, "float32": np.float32, "float": np.float32,
    "float64": np.float64, "double": np.float64, "bfloat16": np.uint16, "int64": np.int64, "long": np.int64,
    "int32": np.int32, "int": np.int32, "int16": np.int16, "int8": np.int8, "uint8": np.uint8, "bool": np.bool_,
}


class Stub:
    """Inert stand-in for any class named in the pickle (one dynamic subclass per qualified name)."""

    _qual = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__dict__.update(args=args, state={}, items=[])
        return obj

    def __init__(self, *args, **kwargs):  # REDUCE path: cls(*args)
        pass

    # pickle protocol hooks ------------------------------------------------
    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = {**(state[0] or {}), **state[1]}  # (dict, slotstate)
        if isinstance(state, dict):
            self.state.update(state)
        else:
            self.state["_state"] = state

    def __setitem__(self, k, v):  # dict subclasses (SETITEMS)
        self.state[k] = v

    def append(self, v):  # list subclasses (APPENDS)
        self.items.append(v)

    def extend(self, vs):
        self.items.extend(vs)

    # convenience ------------------------------------------------------------
    @property
    def cls(self) -> str:
        return type(self)._qual

    @property
    def name(self) -> str:
        return self.cls.rsplit(".", 1)[-1]

    def __getattr__(self, k):
        st = self.__dict__.get("state", {})
        if k in st:
            return st[k]
        raise AttributeError(f"{self.cls} has no attribute {k!r}")

    def get(self, k, default=None):
        return self.state.get(k, default)

    @property
    def children(self) -> "collections.OrderedDict[str, Stub]":
        m = self.state.get("_modules")
        return m if m is not None else collections.OrderedDict()

    def param(self, name: str):
        p = self.state.get("_parameters") or {}
        if p.get(name) is not None:
            return p[name]
        return (self.state.get("_buffers") or {}).get(name)

    def __repr__(self):
        return f"<Stub {self.name}>"


_STUB_TYPES: Dict[str, type] = {}


def _stub_type(qualname: str) -> type:
    t = _STUB_TYPES.get(qualname)
    if t is None:
        t = type(qualname.rsplit(".", 1)[-1], (Stub,), {"_qual": qualname})
        _STUB_TYPES[qualname] = t
    return t


def _rebuild_tensor(storage, offset, size, stride, *rest):
    arr, = storage if isinstance(storage, tuple) else (storage,)
    itemsize = arr.itemsize
    if not size:
        return arr[offset].copy()
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()


def _rebuild_parameter(data, *rest):
    return data


class _SafeUnpickler(pickle.Unpickler):
    _REAL = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("builtins", "set"): set, ("builtins", "frozenset"): frozenset, ("builtins", "slice"): slice,
        ("builtins", "complex"): complex,
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
        ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
        ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
        ("torch._utils", "_rebuild_parameter_with_state"): _rebuild_parameter,
    }

    def __init__(self, fileobj, zf: zipfile.ZipFile, prefix: str):
        super().__init__(fileobj)
        self._zf, self._prefix = zf, prefix
        self._cache: Dict[str, np.ndarray] = {}

    def find_class(self, module: str, name: str):
        real = self._REAL.get((module, name))
        if real is not None:
            return real
        if module == "torch" and name in _TORCH_DTYPES:
            return _stub_type(f"torch.{name}")()
        if module == "torch" and name.endswith("Storage"):
            return name  # storage type marker used in persistent ids
        return _stub_type(f"{module}.{name}")

    def persistent_load(self, pid: Tuple[Any, ...]):
        kind, stype, key, _location, numel = pid[:5]
        if kind != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {kind!r}")
        if key in self._cache:
            return self._cache[key]
        if isinstance(stype, Stub):  # torch >= 2: storage type given as a dtype object
            dtype = _TORCH_DTYPES.get(stype.name, np.float32)
        else:
            dtype = _STORAGE_DTYPES.get(str(stype), np.float32)
        raw = self._zf.read(f"{self._prefix}data/{key}")
        arr = np.frombuffer(raw, dtype=dtype)
        if str(stype) == "BFloat16Storage" or (isinstance(stype, Stub) and stype.name == "bfloat16"):
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        self._cache[key] = arr
        return arr


def load_pt(path) -> Any:
    """Load a torch zip-format checkpoint into plain Python / numpy / Stub objects."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]
        with zf.open(pkl) as f:
            return _SafeUnpickler(io.BytesIO(f.read()), zf, prefix).load()


def describe(obj: Any, depth: int = 0, max_depth: int = 3) -> str:
    """Pretty tree for debugging a checkpoint."""
    pad = "  " * depth
    if isinstance(obj, Stub):
        lines = [f"{pad}{obj.name}"]
        if depth < max_depth:
            for k, v in obj.children.items():
                lines.append(f"{pad}  [{k}]")
                lines.append(describe(v, depth + 2, max_depth))
        return "\n".join(lines)
    if isinstance(obj, np.ndarray):
        return f"{pad}array{obj.shape} {obj.dtype}"
    return f"{pad}{type(obj).__name__}"
