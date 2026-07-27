#!/usr/bin/env python3
"""血袋 OCR RKNN Web 服务(请求-响应模式)。

POST /api/ocr  上传图片(multipart, 字段名 image) -> JSON {ok, elapsed_ms, results}
GET  /         上传测试页

启动: python3 rknn_server.py  (监听 0.0.0.0:8000)
"""
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from rknn_core.rapid_rknn import RapidRKNN

MODEL_DIR = "/root/blood_ocr_rknn/models"
DICT_PATH = "/root/blood_ocr_rknn/rknn_core/ppocr_keys_v1.txt"

app = Flask(__name__)
ocr = RapidRKNN(MODEL_DIR, DICT_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ocr", methods=["POST"])
def ocr_api():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "未提供 image 文件"}), 400
    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "无法解码图片"}), 400

    t = time.time()
    result = ocr(img)
    elapsed = round((time.time() - t) * 1000, 1)

    return jsonify(
        {
            "ok": True,
            "elapsed_ms": elapsed,
            "results": [
                {"text": text, "confidence": round(conf, 4), "box": box}
                for box, text, conf in result
            ],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
