"""rec: 文本识别(CRNN + CTC)。输入固定 48x320, 输出 [N, seq, 6625]。"""
import math

import cv2
import numpy as np

REC_H, REC_W = 48, 320


def preprocess(img):
    """resize 到 48 高(按比例, 不超 320 宽), pad 到 320, 归一化。返回 [1,3,48,320]。"""
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = min(int(math.ceil(REC_H * ratio)), REC_W)
    resized = cv2.resize(img, (resized_w, REC_H))
    resized = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    canvas = np.zeros((3, REC_H, REC_W), dtype=np.float32)
    canvas[:, :, :resized_w] = resized
    return canvas[np.newaxis]


class CTCDecoder:
    """CTC 解码: 字典开头插 blank, 末尾插 space(与 rapidocr 一致)。"""

    def __init__(self, dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            chars = [line.rstrip("\n").rstrip("\r\n") for line in f]
        chars.insert(0, "blank")
        chars.append(" ")
        self.chars = chars

    def decode(self, preds):
        """preds: [N, seq, num_chars] -> List[(text, conf)]。"""
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        results = []
        for i in range(preds_idx.shape[0]):
            idx = preds_idx[i]
            prob = preds_prob[i]
            selection = np.ones(len(idx), dtype=bool)
            selection[1:] = idx[1:] != idx[:-1]
            selection &= idx != 0  # 去 blank
            text = "".join(self.chars[j] for j in idx[selection])
            conf = float(np.mean(prob[selection])) if selection.any() else 0.0
            results.append((text, conf))
        return results
