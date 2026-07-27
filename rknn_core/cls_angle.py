"""cls: 方向分类。输入固定 48x192, 输出 [N,2]。"""
import math

import cv2
import numpy as np

CLS_H, CLS_W = 48, 192
CLS_THRESH = 0.9
LABEL_LIST = ["0", "180"]


def preprocess(img):
    """resize 到 48 高(按比例, 不超 192 宽), pad 到 192, 归一化。返回 [1,3,48,192]。"""
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = min(int(math.ceil(CLS_H * ratio)), CLS_W)
    resized = cv2.resize(img, (resized_w, CLS_H))
    resized = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    canvas = np.zeros((3, CLS_H, CLS_W), dtype=np.float32)
    canvas[:, :, :resized_w] = resized
    return canvas[np.newaxis]


def postprocess_batch(prob):
    """prob: [N,2] -> List[(label, score)]。"""
    idx = prob.argmax(axis=1)
    scores = prob.max(axis=1)
    return [(LABEL_LIST[i], float(s)) for i, s in zip(idx, scores)]
