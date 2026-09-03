# -*- coding: utf-8 -*-
"""推理入口：用检测引擎对图片/视频检测工业设施缺陷并叠加标注。

用法:
    python scripts/detect.py --engine engines/rail-yolo11 \
        --weights <best.pt> --source <图片/视频/0(摄像头)>
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="engines/rail-yolo11")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0", help="图片/视频路径或摄像头索引")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", action="store_true", default=True)
    args = ap.parse_args()

    engine = os.path.abspath(args.engine)
    if not os.path.isdir(os.path.join(engine, "ultralytics")):
        print(f"[x] 未找到检测引擎: {engine}")
        return 1
    sys.path.insert(0, engine)
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[x] 依赖缺失: {e}")
        return 1
    if not os.path.exists(args.weights):
        print(f"[x] 权重不存在: {args.weights}")
        return 1

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        project="runs/detect",
        name="industrial-defect",
    )
    print("[done] 推理完成，结果在 runs/detect/industrial-defect")
    return 0


if __name__ == "__main__":
    sys.exit(main())