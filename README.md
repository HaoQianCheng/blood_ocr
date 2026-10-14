[中文](README.md) | [English](README_EN.md)

# blood_ocr

面向血袋标签图片的 OCR 实验项目，提供 PC 端 RapidOCR/ONNX 推理、RK3576 板端 RKNN 推理，以及图片上传测试页面和 HTTP 接口。另包含方向校正、半自动血型标注、识别结果预览和模型转换脚本。

当前推理输出为文字、置信度和文本框；没有完整的标签字段结构化或医学判定流程。OCR 结果需要人工核对，不应用作临床或输血决策依据。本文依据仓库源码整理，未完成依赖安装、模型推理或板端实测。

## 运行方式

| 方式 | 入口 | 运行环境 | 输出 |
| --- | --- | --- | --- |
| PC OCR | `run.py` | Python + `rapidocr_onnxruntime` | 终端文字、置信度 |
| 板端 OCR | `rknn_run.py` | RK3576 + RKNNLite | 终端文字、置信度 |
| 板端 Web/API | `rknn_server.py` | 同上，另需 Flask | 上传页面、JSON 结果 |

PC 入口调用 `RapidOCR()`，不使用仓库内的 `.rknn` 模型。板端入口通过 `rknn_core/` 加载 `models/det.rknn`、`cls.rknn` 和 `rec.rknn`，依次执行文本检测、方向分类和文字识别。

## 目录结构

```text
blood_ocr/
├── run.py                    # PC RapidOCR 入口
├── rknn_run.py               # 板端命令行入口
├── rknn_server.py            # 板端 Flask 服务
├── requirements.txt         # 现有 Paddle 依赖清单，与 PC 入口不一致
├── requirements-rknn.txt    # 板端依赖清单，不含 Flask
├── rknn_core/               # RKNNLite 引擎及 det/cls/rec 前后处理
├── models/                  # 三个 RKNN 模型
├── templates/index.html     # 图片上传测试页面
├── export/                  # ONNX 检查、字典提取及 RKNN 转换
│   └── ppocr_keys_v1.txt     # 已提交的识别字典
├── train/                   # 数据整理、人工标注、预览工具
└── output/                  # 已提交的裁剪图片
```

`test_images/` 和 `data/` 未随当前源码提供；请自行准备有权使用且已脱敏的测试图片。本文不展示仓库中的标签图片。

## PC：RapidOCR / ONNX

### 准备环境

需要 Python、pip 和虚拟环境支持。仓库没有声明或锁定 PC 端 Python/RapidOCR 兼容版本，以下为依据代码导入列出的安装起点，并非已验证的锁定环境。

```bash
git clone https://github.com/HaoQianCheng/blood_ocr.git
cd blood_ocr
python -m venv .venv
```

激活虚拟环境：Linux/macOS 使用 `source .venv/bin/activate`；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。

```bash
python -m pip install rapidocr-onnxruntime Pillow
```

注意：现有 `requirements.txt` 仅列出 `paddlepaddle==2.6.2`、`paddleocr==2.9.1` 和 `opencv-python-headless`，没有 `run.py` 及 `train/` 使用的 `rapidocr_onnxruntime`。仅执行 `pip install -r requirements.txt` 不能满足当前 PC 入口的依赖。这里没有修改该依赖文件，也未验证上述安装组合。

### 运行

从仓库根目录执行，替换为自己的图片路径：

```bash
python run.py /path/to/image.jpg
python run.py /path/to/images
```

不传参数时读取 `test_images/`。目录模式仅遍历一层，筛选 `.jpg`、`.jpeg`、`.png`、`.bmp`（扩展名不区分大小写）。结果打印到终端，不自动保存 JSON 或标注图片；没有检测到文字时输出提示。

## RK3576：板端 CLI 与 Web/API

### 环境与模型

`requirements-rknn.txt` 的注释指定 RK3576、Python 3.10，并固定 `rknn-toolkit-lite2==2.3.2`。需要与模型和运行库兼容的板端系统及 NPU 驱动；普通 PC 环境不能直接替代板端推理环境。本仓库未提供完整系统镜像或驱动安装流程。

在板端的独立 Python 环境内执行：

```bash
python3 -m pip install -r requirements-rknn.txt
# 只有 Web/API 模式需要；现有清单未包含 Flask
python3 -m pip install Flask
```

### 先修改固定路径

`rknn_run.py` 和 `rknn_server.py` 都使用以下固定路径，启动前必须分别核对并调整：

```python
MODEL_DIR = "/root/blood_ocr_rknn/models"
DICT_PATH = "/root/blood_ocr_rknn/rknn_core/ppocr_keys_v1.txt"
```

- `MODEL_DIR` 应指向三个 `.rknn` 文件所在目录。
- 字典实际在 `export/ppocr_keys_v1.txt`，不在 `rknn_core/`。例如仓库位于 `/root/blood_ocr_rknn` 时，将两个入口的 `DICT_PATH` 都改为 `/root/blood_ocr_rknn/export/ppocr_keys_v1.txt`。
- 若使用其他目录，两个路径都应替换为实际的绝对路径。代码没有模型路径命令行选项。

### 命令行

```bash
python3 rknn_run.py /path/to/image.jpg
python3 rknn_run.py /path/to/images
```

默认目录、目录遍历方式和扩展名筛选与 PC 入口一致。读取失败的图片会被跳过；正常处理结束后释放模型资源。

### Web 服务与 API

```bash
python3 rknn_server.py
```

服务默认监听 `0.0.0.0:8000`，访问 `http://<板端IP>:8000/` 可打开上传页面。模型在服务导入阶段加载，因此模型、字典或运行库问题会阻止服务启动。

使用无个人信息的测试图片调用接口：

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "image=@/path/to/image.jpg"
```

请求为 `multipart/form-data`，文件字段名必须为 `image`。成功响应包含：

- `ok`：`true`
- `elapsed_ms`：OCR 调用耗时（毫秒），不含上传和图片解码时间
- `results`：数组，每项含 `text`、`confidence`、`box`（原图上的四点坐标）

没有识别结果时，`results` 为空数组。缺少 `image` 或图片无法解码时返回 HTTP 400，含 `ok: false` 和 `error`。

当前使用 Flask 开发服务器，未实现认证、TLS、上传大小限制或生产部署配置。默认会监听所有网卡；仅在可信测试网络中使用，正式部署前需补齐访问控制和安全配置。不要向未受保护的服务上传可识别个人身份或含敏感信息的图片。

## 数据辅助工具（PC）

以下工具使用 RapidOCR 和 Pillow，建议在 PC 环境运行：

```bash
python train/fix_orientation.py /path/to/images data/aligned
python train/preview_ocr.py data/aligned
python train/label.py data/aligned data/labels.txt
```

- `fix_orientation.py`：尝试 0/90/180/270 度，通过 OCR 结果及关键词评分选择方向；默认 `output/` → `data/aligned/`。
- `preview_ocr.py`：绘制文本框、文字和置信度，固定输出到 `data/preview/`；默认输入为 `output/`。字体优先使用 `msyh.ttc`，其他系统可能需要改为可用的中文字体。
- `label.py`：通过图片查看器和终端交互确认或修正 A/B/O/AB 标签，输出 `文件名<TAB>血型`。需要图形图片查看器。保存时会重写标注文件；提前退出可能漏掉尚未遍历到的已有标注，运行前请备份文件。

`train/` 当前是数据辅助脚本目录，没有提供模型训练入口或训练配置。

## 模型检查与转换（可选）

仓库已有 RKNN 模型，运行板端推理不需要先重新转换。需要重建模型时，先准备兼容的 Linux/WSL 转换环境、RapidOCR ONNX 模型、`onnx` 和提供 `rknn.api` 的 RKNN-Toolkit2；它与板端的 RKNN-Toolkit-Lite2 用途不同。

- `export/inspect_onnx.py`：检查三个 ONNX 模型的输入输出，并从识别模型 metadata 提取字典。先修改 `MODELS_DIR` 与 `DICT_OUT`。
- `export/onnx2rknn.py`：先修改 `ONNX_DIR`、`OUT_DIR`；当前 `TARGET = "rk3576"`，关闭量化，检测模型配置四种动态输入尺寸，分类和识别模型使用固定尺寸。
- `export/download_rknn.py`：下载指定的 Toolkit2 2.3.2、CPython 3.10、x86_64 wheel 到 `/tmp/rknn_toolkit2.whl`，不执行安装。链接可用性、文件名及平台兼容性需另行核对。

转换脚本内含作者机器上的绝对路径，不应直接照搬。完成环境准备和路径调整后，从仓库根目录运行 `python3 export/inspect_onnx.py` 与 `python3 export/onnx2rknn.py`。此处未验证转换或生成模型的兼容性。

## 已知限制

- 依赖清单存在上述缺项和入口不一致问题；没有统一的已验证安装流程。
- RKNN 管线固定识别置信度阈值为 `0.6`，分类/识别逐框 batch=1 推理；与 PC RapidOCR 的输出不保证一致。
- 未提供自动化测试、准确率评测或可靠的性能基准。置信度不是医学正确性的保证。
- 辅助脚本的路径、字体和 Windows cuDNN 路径处理带有本机环境假设，跨平台使用需要检查。
- 当前仓库未提供许可证文件；代码、模型与数据的使用及再分发权限需分别确认。
