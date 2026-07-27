"""检查 rapidocr 的 3 个 ONNX 模型: 输入输出 shape + opset + 提取 rec 字典。

在 WSL 里运行:
    python3 /mnt/c/Users/qianc/Projects/blood_ocr/export/inspect_onnx.py
"""
from pathlib import Path

import onnx

MODELS_DIR = Path(
    "/usr/local/lib/python3.10/dist-packages/rapidocr_onnxruntime/models"
)
MODELS = {
    "det": "ch_PP-OCRv4_det_infer.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "rec": "ch_PP-OCRv4_rec_infer.onnx",
}
# 字典输出到项目 export 目录(WSL 通过 /mnt/c 访问 Windows 侧)
DICT_OUT = Path("/mnt/c/Users/qianc/Projects/blood_ocr/export/ppocr_keys_v1.txt")


def dims_of(value_info):
    return [
        d.dim_value if d.dim_value else d.dim_param
        for d in value_info.type.tensor_type.shape.dim
    ]


for key, name in MODELS.items():
    m = onnx.load(str(MODELS_DIR / name))
    print(f"\n=== {key}: {name} ===")
    for inp in m.graph.input:
        print(f"  IN  {inp.name}: {dims_of(inp)}")
    for out in m.graph.output:
        print(f"  OUT {out.name}: {dims_of(out)}")
    print(f"  opset: {[o.version for o in m.opset_import]}, ir: {m.ir_version}")

# rec 字典(内嵌在 onnx metadata)
rec = onnx.load(str(MODELS_DIR / MODELS["rec"]))
meta = {p.key: p.value for p in rec.metadata_props}
print(f"\n=== rec metadata keys: {list(meta.keys())}")
if "character" in meta:
    chars = meta["character"].splitlines()
    print(f"字典长度: {len(chars)}")
    print(f"  前10: {chars[:10]}")
    print(f"  后10: {chars[-10:]}")
    DICT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DICT_OUT.write_text("\n".join(chars) + "\n", encoding="utf-8")
    print(f"字典已保存: {DICT_OUT}")
