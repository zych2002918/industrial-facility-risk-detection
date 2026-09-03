# -*- coding: utf-8 -*-
"""数据准备引导：下载/转换公开替代数据集为 YOLO 格式。

- 不下载生产数据（协议保护）。
- 支持 NEU-DET 等公开数据的组织与 train/val 划分；其余数据集给出手工指引。

用法:
    python scripts/prepare_data.py --dataset neu-det --root ./datasets
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

YOLO_NAMES_NEU = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def prepare_neu(root: Path) -> None:
    """NEU-DET 组织说明：要求用户先下载 Images 与 annotations 到 datasets/neu-det-raw。

    若检测到已下载的 NEU-DET 目录，则按 8:1:1 划分 train/val/test 并生成
    images 与 labels 目录（标注为 XML，需要 xml2txt 转换——可复用
    rail-surface-defect-yolo11 仓库的 dataset/xml2txt.py 逻辑）。
    """
    raw = root / "neu-det-raw"
    if not raw.exists():
        print("[i] 未找到 datasets/neu-det-raw，请先下载 NEU-DET：")
        print("    http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/")
        print("    或 kaggle 镜像 'NEU-DET'（Images + ANNOTATIONS），解压到: " + str(raw))
        return
    imgs = raw / "IMAGES" if (raw / "IMAGES").exists() else raw / "images"
    anns = raw / "ANNOTATIONS" if (raw / "ANNOTATIONS").exists() else raw / "annotations"
    if not imgs.exists() or not anns.exists():
        print("[i] NEU-DET 解压结构需含 IMAGES/ 与 ANNOTATIONS/ 子目录")
        return
    out = root / "industrial" / "images"
    for split, frac in (("train", 0.8), ("valid", 0.1)):
        (out / split).mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    # 注：完整 xml->txt 转换 + 划分逻辑在此展开（此处为骨架占位）
    print(f"[i] NEU-DET 布局已确认：{len(list(imgs.glob('*.jpg')))} 张图像 -> 按 8:1:1 划分")
    print("    完整转换请参照 rail-surface-defect-yolo11/dataset/xml2txt.py 实现后填入本脚本")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["neu-det", "manual"], default="neu-det")
    ap.add_argument("--root", default="./datasets")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.dataset == "neu-det":
        prepare_neu(root)
    else:
        print("[i] manual 模式：请按 data/defects_sample.yaml 注释指引手工准备数据")
    print("[done] 数据准备说明完毕")
    return 0


if __name__ == "__main__":
    sys.exit(main())