"""
在图片上画 OCR 识别框和文字，保存到 data/preview/
用法: python train/preview_ocr.py [图片目录]
"""
import os
import sys
from pathlib import Path

# 设置 cuDNN 路径
_cudnn_bin = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
if _cudnn_bin.exists():
    os.environ["PATH"] = str(_cudnn_bin) + ";" + os.environ.get("PATH", "")

from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR


def draw_result(img, result):
    """在图片上画识别框和文字"""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("msyh.ttc", 20)
    except:
        font = ImageFont.load_default()

    if not result:
        return img

    for box, text, conf in result:
        # 画框（box 是四点坐标）
        points = [(int(p[0]), int(p[1])) for p in box]
        draw.polygon(points, outline="red", width=2)

        # 画文字标签
        x, y = int(box[0][0]), max(int(box[0][1]) - 25, 0)
        label = f"{text} ({conf:.0%})"
        draw.text((x, y), label, fill="red", font=font)

    return img


def main():
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    out_dir = Path("data/preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in image_exts)

    if not images:
        print(f"目录 {in_dir} 下没有图片")
        return

    ocr = RapidOCR()
    print(f"共 {len(images)} 张图片，开始识别并标注...\n")

    for img_path in images:
        img = Image.open(img_path)
        result, _ = ocr(img)

        annotated = draw_result(img.copy(), result)
        annotated.save(out_dir / img_path.name)

        # 打印识别结果
        print(f"  {img_path.name}:")
        if result:
            for _, text, conf in result:
                print(f"    {text:<30s} {conf:.2%}")
        else:
            print(f"    未识别到文字")
        print()

    print(f"标注图片保存到 {out_dir}/")


if __name__ == "__main__":
    main()
