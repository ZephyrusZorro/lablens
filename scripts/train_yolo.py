"""Fine-tune a component detector on photos of YOUR kit (the biggest accuracy win for a live demo).

    pip install -r requirements-train.txt
    python scripts/train_yolo.py --data path/to/data.yaml --epochs 80

data.yaml is the standard Ultralytics format (Roboflow "YOLOv8" export works as-is). Start from
150-300 top-down photos of your bench under your lab lighting; label them in Roboflow or CVAT.
The best checkpoint is copied to models/my_kit.pt. Add it to config.yaml under vision.models and
LabLens converts it to ONNX (torch-free) on first load and fuses it with the bundled detectors.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--base", default="yolov8n.pt", help="yolov8n.pt (fast) or yolov8s.pt (more accurate)")
p.add_argument("--epochs", type=int, default=80)
p.add_argument("--imgsz", type=int, default=800)
p.add_argument("--out", default="models/my_kit.pt")
a = p.parse_args()

model = YOLO(a.base)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=16, patience=20, project="runs", name="lablens",
            exist_ok=True)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(model.trainer.best, out)
metrics = model.val(data=a.data, imgsz=a.imgsz)
print(f"\nsaved {out}   mAP@0.5 = {metrics.box.map50:.3f}   mAP@0.5:0.95 = {metrics.box.map:.3f}")
print("Add to config.yaml:\n  vision:\n    models:\n      - path: %s\n        imgsz: %d\n        weight: 1.5\n" % (out, a.imgsz))
