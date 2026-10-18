"""Convert Ultralytics YOLO detection checkpoints (.pt) to ONNX - without PyTorch.

Supported blocks: Conv (+BN folded), DWConv, Conv2d, C2f, C3, C3k2/C3k, Bottleneck, SPPF, SPP,
Upsample, Concat, Detect. That covers YOLOv8 and YOLOv5u detect models (n/s/m/l/x). Unsupported
blocks raise a clear error naming the layer.

Graph output: `raw` of shape (batch, 4*reg_max + nc, anchors) in Ultralytics' native layout
(un-decoded DFL logits + class logits). Decoding is done in numpy (lablens.vision.yolo_onnx), which
keeps the graph portable and lets us skip work on low-confidence anchors.

    python -m lablens.vision.export models/best.pt models/best.onnx
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lablens.vision import onnx_writer as ow
from lablens.vision.ptload import Stub, load_pt

log = logging.getLogger(__name__)


class UnsupportedLayer(RuntimeError):
    pass


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


class YoloOnnxExporter:
    def __init__(self, model: Stub):
        self.m = model
        self.g = ow.GraphBuilder()

    # ------------------------------------------------------------------ primitives
    def conv2d(self, x: str, c: Stub, bn: Optional[Stub] = None) -> str:
        w = c.param("weight").astype(np.float64)
        b = c.param("bias")
        b = np.zeros(w.shape[0]) if b is None else b.astype(np.float64)
        if bn is not None:  # fold BatchNorm into the convolution
            gamma, beta = bn.param("weight").astype(np.float64), bn.param("bias").astype(np.float64)
            mean, var = bn.param("running_mean").astype(np.float64), bn.param("running_var").astype(np.float64)
            scale = gamma / np.sqrt(var + float(bn.get("eps", 1e-3)))
            w = w * scale[:, None, None, None]
            b = (b - mean) * scale + beta
        k = tuple(c.get("kernel_size"))
        s, p, d = tuple(c.get("stride")), tuple(c.get("padding")), tuple(c.get("dilation", (1, 1)))
        if isinstance(c.get("padding"), str):
            raise UnsupportedLayer("string padding")
        return self.g.op("Conv", [x, self.g.init(_f32(w)), self.g.init(_f32(b), "b")],
                         kernel_shape=list(k), strides=list(s), pads=[p[0], p[1], p[0], p[1]],
                         dilations=list(d), group=int(c.get("groups", 1)))

    def act(self, x: str, a: Optional[Stub]) -> str:
        if a is None:
            return x
        n = a.name
        if n == "Identity":
            return x
        if n == "SiLU":
            return self.g.op("Mul", [x, self.g.op("Sigmoid", [x])])
        if n == "ReLU":
            return self.g.op("Relu", [x])
        if n == "LeakyReLU":
            return self.g.op("LeakyRelu", [x], alpha=float(a.get("negative_slope", 0.01)))
        if n == "Hardswish":
            return self.g.op("HardSwish", [x])
        raise UnsupportedLayer(f"activation {n}")

    def concat(self, xs: List[str], axis: int = 1) -> str:
        return xs[0] if len(xs) == 1 else self.g.op("Concat", xs, axis=axis)

    def split(self, x: str, sizes: List[int]) -> List[str]:
        return self.g.op("Split", [x, self.g.const(np.array(sizes, dtype=np.int64))], n_out=len(sizes), axis=1)

    # ------------------------------------------------------------------ modules
    def module(self, x, mod: Stub):
        n = mod.name
        ch = mod.children
        if n in ("Conv", "DWConv"):
            return self.act(self.conv2d(x, ch["conv"], ch.get("bn")), ch.get("act"))
        if n == "Conv2d":
            return self.conv2d(x, mod)
        if n == "Sequential":
            for sub in ch.values():
                x = self.module(x, sub)
            return x
        if n == "Bottleneck":
            y = self.module(self.module(x, ch["cv1"]), ch["cv2"])
            return self.g.op("Add", [x, y]) if mod.get("add", False) else y
        if n in ("C2f", "C3k2"):
            y1 = self.module(x, ch["cv1"])
            c = int(mod.get("c"))
            ys = list(self.split(y1, [c, c]))
            for b in ch["m"].children.values():
                ys.append(self.module(ys[-1], b))
            return self.module(self.concat(ys), ch["cv2"])
        if n in ("C3", "C3k"):
            a = self.module(x, ch["cv1"])
            for b in ch["m"].children.values():
                a = self.module(a, b)
            return self.module(self.concat([a, self.module(x, ch["cv2"])]), ch["cv3"])
        if n == "SPPF":
            y = [self.module(x, ch["cv1"])]
            mp = ch["m"]
            k, p = int(mp.get("kernel_size")), int(mp.get("padding"))
            for _ in range(int(mod.get("n", 3))):
                y.append(self.g.op("MaxPool", [y[-1]], kernel_shape=[k, k], strides=[1, 1], pads=[p, p, p, p]))
            out = self.module(self.concat(y), ch["cv2"])
            return self.g.op("Add", [out, x]) if mod.get("add", False) else out
        if n == "SPP":
            a = self.module(x, ch["cv1"])
            ys = [a]
            for pool in ch["m"].children.values():
                k, p = int(pool.get("kernel_size")), int(pool.get("padding"))
                ys.append(self.g.op("MaxPool", [a], kernel_shape=[k, k], strides=[1, 1], pads=[p, p, p, p]))
            return self.module(self.concat(ys), ch["cv2"])
        if n == "Upsample":
            if mod.get("mode", "nearest") != "nearest" or mod.get("size") is not None:
                raise UnsupportedLayer("Upsample other than nearest/scale_factor")
            sf = float(mod.get("scale_factor"))
            scales = self.g.const(np.array([1, 1, sf, sf], dtype=np.float32))
            return self.g.op("Resize", [x, "", scales], mode="nearest",
                             coordinate_transformation_mode="asymmetric", nearest_mode="floor")
        if n == "Concat":
            return self.concat(x, int(mod.get("d", 1)))
        if n == "Identity":
            return x
        raise UnsupportedLayer(f"{n} (layer type {mod.get('type', '?')}) is not supported by the exporter")

    def detect(self, xs: List[str], det: Stub) -> str:
        if det.name not in ("Detect",):
            raise UnsupportedLayer(f"head {det.name}: only plain Detect heads are supported")
        nc, reg = int(det.get("nc")), int(det.get("reg_max", 16))
        no = nc + 4 * reg
        shape = self.g.const(np.array([0, no, -1], dtype=np.int64))
        outs = []
        for i, x in enumerate(xs):
            box = self.module(x, det.children["cv2"].children[str(i)])
            cls = self.module(x, det.children["cv3"].children[str(i)])
            outs.append(self.g.op("Reshape", [self.concat([box, cls]), shape]))
        return self.g.op("Concat", outs, axis=2)

    # ------------------------------------------------------------------ whole model
    def build(self) -> bytes:
        layers = list(self.m.children["model"].children.values())
        y: List[Optional[str]] = []
        x = "images"
        for li, layer in enumerate(layers):
            f = layer.get("f", -1)
            if f != -1:
                x = [x if j == -1 else y[j] for j in f] if isinstance(f, list) else y[f]
            if li == len(layers) - 1:
                x = self.detect(x if isinstance(x, list) else [x], layer)
            else:
                x = self.module(x, layer)
            y.append(x)
        det = layers[-1]
        names = self.m.get("names") or {i: str(i) for i in range(int(det.get("nc")))}
        meta = {
            "names": json.dumps({int(k): v for k, v in dict(names).items()}),
            "stride": json.dumps([float(s) for s in np.asarray(det.get("stride")).ravel()]),
            "reg_max": str(int(det.get("reg_max", 16))),
            "nc": str(int(det.get("nc"))),
            "task": "detect",
            "exporter": "lablens torch-free exporter",
        }
        no = int(det.get("nc")) + 4 * int(det.get("reg_max", 16))
        return ow.model(self.g.nodes, self.g.inits,
                        [ow.value_info("images", ow.FLOAT, ["batch", 3, "height", "width"])],
                        [ow.value_info(x, ow.FLOAT, ["batch", no, "anchors"])],
                        graph_name="lablens_yolo", metadata=meta)


def export(pt_path, onnx_path, extra_meta: Optional[Dict[str, str]] = None) -> Dict:
    ck = load_pt(pt_path)
    model = ck.get("ema") or ck.get("model") if isinstance(ck, dict) else ck
    if model is None:
        raise ValueError(f"{pt_path}: no model in checkpoint")
    exp = YoloOnnxExporter(model)
    blob = exp.build()
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    Path(onnx_path).write_bytes(blob)
    info = {
        "names": {int(k): v for k, v in dict(model.get("names")).items()},
        "train_metrics": ck.get("train_metrics") if isinstance(ck, dict) else None,
        "train_args": {k: (ck.get("train_args") or {}).get(k) for k in ("model", "epochs", "imgsz")}
        if isinstance(ck, dict) and isinstance(ck.get("train_args"), dict) else None,
        "ultralytics_version": ck.get("version") if isinstance(ck, dict) else None,
    }
    if extra_meta:
        info.update(extra_meta)
    Path(str(onnx_path) + ".json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Torch-free YOLO .pt -> .onnx converter")
    ap.add_argument("pt")
    ap.add_argument("onnx")
    a = ap.parse_args()
    print(json.dumps(export(a.pt, a.onnx), indent=2, default=str))
