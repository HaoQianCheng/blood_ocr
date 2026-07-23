"""
标签图片方向校正脚本
对每张图片尝试 4 个方向，选择 OCR 置信度最高的方向
用法: python train/fix_orientation.py [输入目录] [输出目录]
"""
import os
import sys
from pathlib import Path

# 设置 cuDNN 路径（nvidia-cudnn-cu12 安装后）
_cudnn_bin = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
if _cudnn_bin.exists():
    os.environ["PATH"] = str(_cudnn_bin) + ";" + os.environ.get("PATH", "")

from PIL import Image
from rapidocr_onnxruntime import RapidOCR


# 血袋标签关键词（正确方向更容易识别出这些词）
KEYWORDS = ["血", "型", "全血", "日期", "生产", "失效", "批号", "血液", "保存", "有效期", "公司"]


def score_result(result):
    """计算 OCR 结果的得分：文字数量 × 平均置信度 + 关键词加分"""
    if not result:
        return 0
    confs = [conf for _, _, conf in result]
    texts = [text for _, text, _ in result]
    all_text = "".join(texts)

    # 基础分：文字数量 × 平均置信度
    base_score = len(confs) * (sum(confs) / len(confs))

    # 关键词加分：每个匹配的关键词 +2.0
    keyword_score = sum(2.0 for kw in KEYWORDS if kw in all_text)

    return base_score + keyword_score


def find_best_orientation(ocr, img):
    """尝试 4 个方向，返回最佳旋转角度"""
    best_angle = 0
    best_score = -1

    for angle in (0, 90, 180, 270):
        rotated = img.rotate(-angle, expand=True) if angle else img
        result, _ = ocr(rotated)
        score = score_result(result)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def main():
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/aligned")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in image_exts)

    if not images:
        print(f"目录 {in_dir} 下没有图片")
        return

    ocr = RapidOCR()
    print(f"共 {len(images)} 张图片，开始校正方向...\n")

    for img_path in images:
        img = Image.open(img_path)
        angle = find_best_orientation(ocr, img)

        if angle:
            corrected = img.rotate(-angle, expand=True)
            print(f"  {img_path.name}: 旋转 {angle}°")
        else:
            corrected = img
            print(f"  {img_path.name}: 方向正确")

        corrected.save(out_dir / img_path.name)

    print(f"\n校正完成，保存到 {out_dir}/")


if __name__ == "__main__":
    main()
