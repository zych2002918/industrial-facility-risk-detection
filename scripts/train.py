# -*- coding: utf-8 -*-
"""训练入口（薄封装）：调用检测引擎 rail-surface-defect-yolo11 的改进版 ultralytics。

用法:
    python scripts/train.py --engine ../rail-surface-defect-yolo11 \
        --data data/defects_sample.yaml --epochs 100 --imgsz 640 --batch 16

检测引擎须先克隆：
    git clone https://github.com/zych2002918/rail-surface-defect-yolo11.git engines/rail-yolo11
    （然后 --engine engines/rail-yolo11）
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="engines/rail-yolo11",
                    help="rail-surface-defect-yolo11 克隆路径（含改进版 ultralytics）")
    ap.add_argument("--data", required=True, help="数据集 yaml（YOLO 格式）")
    ap.add_argument("--cfg", default="ultralytics/cfg/models/gaijin11/LSDECD-FDPN-ODConv.yaml",
                    help="模型结构 yaml（相对 engine 根；默认用改进模型）")
    ap.add_argument("--weights", default=None, help="预训练权重（可选，如 rail 仓库 best.pt 做迁移）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    engine = os.path.abspath(args.engine)
    if not os.path.isdir(os.path.join(engine, "ultralytics")):
        print(f"[x] 未找到检测引擎 ultralytics 包: {engine}")
        print("    请先: git clone https://github.com/zych2002918/rail-surface-defect-yolo11.git " + engine)
        return 1
    sys.path.insert(0, engine)  # 使 from ultralytics import YOLO 命中改进包

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[x] 引擎依赖缺失: {e}")
        print("    请先: pip install -r engines/rail-yolo11/requirements.txt")
        return 1

    cfg_path = args.cfg
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(engine, cfg_path)
    if not os.path.exists(cfg_path):
        print(f"[i] 未找到结构 yaml {cfg_path}，回退官方 yolo11.yaml")
        cfg_path = os.path.join(engine, "ultralytics/cfg/models/11/yolo11.yaml")

    model = YOLO(cfg_path)
    if args.weights and os.path.exists(args.weights):
        print(f"[i] 加载预训练权重 {args.weights}")
        model = YOLO(args.weights)

    model.train(
        data=os.path.abspath(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/train",
        name="industrial-defect",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())