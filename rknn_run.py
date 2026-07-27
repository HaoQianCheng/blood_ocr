"""RKNN 版 OCR 推理入口(对标 run.py, 板端运行)。
用法: python3 rknn_run.py [图片路径|目录]
"""
import sys
from pathlib import Path

import cv2

from rknn_core.rapid_rknn import RapidRKNN

MODEL_DIR = "/root/blood_ocr_rknn/models"
DICT_PATH = "/root/blood_ocr_rknn/rknn_core/ppocr_keys_v1.txt"


def main():
    ocr = RapidRKNN(MODEL_DIR, DICT_PATH)

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_images")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in image_exts
        )
    else:
        print(f"路径不存在: {input_path}")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"读取失败: {img_path}")
            continue
        result = ocr(img)
        print(f"\n{'=' * 40}")
        print(f"图片: {img_path.name}")
        print(f"{'=' * 40}")
        if not result:
            print("  未检测到文字")
            continue
        for box, text, conf in result:
            print(f"  {text:<30s} 置信度: {conf:.2%}")

    ocr.release()


if __name__ == "__main__":
    main()
