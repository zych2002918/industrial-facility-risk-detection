# -*- coding: utf-8 -*-
"""边缘部署第一步：PyTorch 权重 -> ONNX。

用法:
    python scripts/export_onnx.py --engine engines/rail-yolo11 \
        --weights <best.pt> --imgsz 640 --half  (可选 --dynamic)

后续平台适配：
  瑞芯微 RK3588 : rknn-toolkit2 (best.onnx -> .rknn, INT8/FP16 量化)
  英伟达 Jetson : trtexec / TensorRT Python API (FP16/INT8)
  通用边缘盒   : onnxruntime (CPU/INT8, 可配合 quantization)
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="engines/rail-yolo11")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="导出 FP16")
    ap.add_argument("--dynamic", action="store_true", help="动态 batch")
    args = ap.parse_args()

    engine = os.path.abspath(args.engine)
    sys.path.insert(0, engine)
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[x] 依赖缺失: {e}")
        return 1

    model = YOLO(args.weights)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=True,
        opset=17,
    )
    print("[done] ONNX 已导出（与权重同目录 best.onnx）")
    return 0


if __name__ == "__main__":
    sys.exit(main())