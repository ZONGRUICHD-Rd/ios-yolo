"""Export Ultralytics YOLO checkpoints to Core ML packages for the iOS app.

Usage: python scripts/export_models.py yolo26n yolo26s --out Models
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="checkpoint names or paths, e.g. yolo26n yolo26s my_model.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--quantize", type=int, default=16, choices=[8, 16, 32], help="weight precision")
    parser.add_argument("--out", default="Models")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        weights = name if name.endswith(".pt") else f"{name}.pt"
        # nms=False keeps YOLO26's NMS-free end-to-end head ([1, 300, 6] output).
        # Older models (YOLO11/v8) export their raw [1, 4+nc, anchors] head; the app runs NMS for those.
        exported = Path(YOLO(weights).export(format="coreml", imgsz=args.imgsz, nms=False, quantize=args.quantize))
        dst = out / exported.name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(exported), dst)
        print(f"exported {weights} -> {dst}")


if __name__ == "__main__":
    main()
