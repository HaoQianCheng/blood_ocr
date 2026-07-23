import sys
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR


def main():
    ocr = RapidOCR()

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_images")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in image_exts)
    else:
        print(f"路径不存在: {input_path}")
        return

    for img_path in images:
        result, _ = ocr(str(img_path))
        print(f"\n{'='*40}")
        print(f"图片: {img_path.name}")
        print(f"{'='*40}")

        if not result:
            print("  未检测到文字")
            continue

        for box, text, confidence in result:
            print(f"  {text:<30s} 置信度: {confidence:.2%}")


if __name__ == "__main__":
    main()
