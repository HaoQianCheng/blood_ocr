"""把 rapidocr 的 3 个 ONNX 模型转成 RKNN (target=rk3576)。

det 动态 shape(fp16), cls/rec fp16(RK3576 NPU 仅支持 fp16)。
在 WSL 里运行: python3 /mnt/c/Users/qianc/Projects/blood_ocr/export/onnx2rknn.py
"""
from pathlib import Path

from rknn.api import RKNN

ONNX_DIR = Path(
    "/usr/local/lib/python3.10/dist-packages/rapidocr_onnxruntime/models"
)
OUT_DIR = Path("/mnt/c/Users/qianc/Projects/blood_ocr/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "rk3576"

DET_DYNAMIC_SHAPES = [
    [[1, 3, 736, 736]],
    [[1, 3, 736, 1280]],
    [[1, 3, 1280, 736]],
    [[1, 3, 1280, 1280]],
]


def _build(rknn, rknn_path):
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"  build 失败: {ret}")
        rknn.release()
        return False
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"  export 失败: {ret}")
        rknn.release()
        return False
    rknn.release()
    print(f"  OK: {rknn_path}")
    return True


def convert_det(onnx_file):
    onnx_path = str(ONNX_DIR / onnx_file)
    rknn_path = str(OUT_DIR / "det.rknn")
    print(f"\n=== det(动态 fp16): {onnx_file} ===")
    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],
        target_platform=TARGET,
        dynamic_input=DET_DYNAMIC_SHAPES,
    )
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  load_onnx 失败: {ret}")
        rknn.release()
        return False
    return _build(rknn, rknn_path)


def convert_fixed(name, onnx_file, input_shape):
    onnx_path = str(ONNX_DIR / onnx_file)
    rknn_path = str(OUT_DIR / f"{name}.rknn")
    print(f"\n=== {name}: {onnx_file} ===")
    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],
        target_platform=TARGET,
    )
    ret = rknn.load_onnx(
        model=onnx_path, inputs=["x"], input_size_list=[input_shape]
    )
    if ret != 0:
        print(f"  load_onnx 失败: {ret}")
        rknn.release()
        return False
    return _build(rknn, rknn_path)


def main():
    ok = 0
    if convert_det("ch_PP-OCRv4_det_infer.onnx"):
        ok += 1
    if convert_fixed("cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx", [1, 3, 48, 192]):
        ok += 1
    if convert_fixed("rec", "ch_PP-OCRv4_rec_infer.onnx", [1, 3, 48, 320]):
        ok += 1
    print(f"\n完成: {ok}/3 个模型转换成功")


if __name__ == "__main__":
    main()
