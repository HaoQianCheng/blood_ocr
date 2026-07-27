"""RapidOCR RKNN 版: det -> cls -> rec 组合 pipeline(板端运行)。

det 动态 shape, cls/rec 固定 batch=1 逐个推理。
text_score=0.6(rec fp16 置信度偏低, 适当抬高过滤误检)。
"""
import cv2
import numpy as np
from rknnlite.api import RKNNLite

from .rknn_engine import RKNNEngine
from . import det_dbnet, cls_angle, rec_crnn


class RapidRKNN:
    def __init__(self, model_dir, dict_path, core_mask=RKNNLite.NPU_CORE_0_1):
        self.det = RKNNEngine(f"{model_dir}/det.rknn", core_mask)
        self.cls = RKNNEngine(f"{model_dir}/cls.rknn", core_mask)
        self.rec = RKNNEngine(f"{model_dir}/rec.rknn", core_mask)
        self.decoder = rec_crnn.CTCDecoder(dict_path)
        self.text_score = 0.6

    def __call__(self, img):
        """img: BGR np.ndarray -> List[(box, text, conf)]。"""
        if img is None:
            raise ValueError("img is None")

        # 1. det 检测文本框(动态 shape)
        det_input, meta = det_dbnet.preprocess(img)
        det_out = self.det.infer(det_input)[0]
        boxes = det_dbnet.postprocess(det_out, img.shape[:2], meta)
        if not boxes:
            return []

        # 2. 透视裁剪每个文本框
        crops = [self._crop(img, b) for b in boxes]

        # 3. cls 方向分类(逐个, batch=1), 180° 则旋转
        for i, crop in enumerate(crops):
            cls_out = self.cls.infer(cls_angle.preprocess(crop))[0]
            label, score = cls_angle.postprocess_batch(cls_out)[0]
            if "180" in label and score > cls_angle.CLS_THRESH:
                crops[i] = cv2.rotate(crop, 1)

        # 4. rec 识别(逐个, batch=1)
        texts = []
        for crop in crops:
            rec_out = self.rec.infer(rec_crnn.preprocess(crop))[0]
            texts.extend(self.decoder.decode(rec_out))

        # 5. 过滤 + 组装
        results = []
        for box, (text, conf) in zip(boxes, texts):
            if conf >= self.text_score:
                results.append((box.tolist(), text, conf))
        return results

    @staticmethod
    def _crop(img, points):
        """透视变换裁剪文本框(照搬 rapidocr get_rotate_crop_image)。"""
        points = points.astype(np.float32)
        w = int(max(np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3])))
        h = int(max(np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2])))
        w, h = max(w, 1), max(h, 1)
        pts_std = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst = cv2.warpPerspective(
            img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )
        if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
            dst = np.rot90(dst)
        return dst

    def release(self):
        self.det.release()
        self.cls.release()
        self.rec.release()
