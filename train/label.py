"""
半自动血型标注脚本
用法：python train/label.py [图片目录] [输出文件]
"""
import sys
import os
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from PIL import Image

# 血型关键词
BLOOD_TYPES = {"A", "B", "O", "AB", "A型", "B型", "O型", "AB型"}


def extract_blood_type(ocr_results):
    """从 OCR 结果中提取血型"""
    if not ocr_results:
        return ""
    for _, text, conf in ocr_results:
        t = text.strip().upper().replace("型", "")
        if t in ("A", "B", "O", "AB"):
            return t
    return ""


def main():
    img_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/aligned")
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/labels.txt")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有标注（支持断点续标）
    labeled = {}
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                name, bt = line.split("\t", 1)
                labeled[name.strip()] = bt.strip()

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in image_exts)

    if not images:
        print(f"目录 {img_dir} 下没有图片")
        return

    ocr = RapidOCR()
    results = []

    print(f"共 {len(images)} 张图片，已有标注 {len(labeled)} 张")
    print("操作：回车=确认  输入=修正  s=跳过  q=退出\n")

    for img_path in images:
        name = img_path.name

        # 已标注的跳过
        if name in labeled:
            results.append(f"{name}\t{labeled[name]}")
            continue

        # OCR 识别
        ocr_result, _ = ocr(str(img_path))
        blood_type = extract_blood_type(ocr_result)

        # 显示图片
        img = Image.open(img_path)
        img.show(title=name)

        # 交互确认
        hint = f"[{blood_type}]" if blood_type else "[未识别]"
        print(f"图片: {name}")
        print(f"OCR 识别血型: {hint}")

        user_input = input("→ 回车确认 / 输入血型(A/B/O/AB) / s跳过 / q退出: ").strip().upper()

        if user_input == "Q":
            break
        if user_input == "S":
            continue

        final = user_input if user_input else blood_type
        if not final:
            print("  ⚠ 未识别到血型，跳过\n")
            continue

        # 校验
        if final not in ("A", "B", "O", "AB"):
            print(f"  ⚠ 无效血型: {final}，跳过\n")
            continue

        results.append(f"{name}\t{final}")
        print(f"  ✅ {name} → {final}\n")

    # 保存
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")

    print(f"\n标注完成，共 {len(results)} 条，保存到 {out_file}")


if __name__ == "__main__":
    main()
