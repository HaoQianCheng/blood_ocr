"""det: 文本检测(DBNet)。动态 shape(4档对齐32), 按原版 resize 逻辑。

前处理: 最短边放大到736, 最长边限1280, 对齐32, 选档位 pad(均值127.5);
后处理: DBNet(照搬 rapidocr DBPostProcess), 坐标映射回原图。
"""
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

DET_SHAPES = [(736, 736), (736, 1280), (1280, 736), (1280, 1280)]
DET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DET_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DET_THRESH = 0.3
DET_BOX_THRESH = 0.5
DET_UNCLIP_RATIO = 1.6
DET_USE_DILATION = True
DET_MAX_CANDIDATES = 1000
DET_MIN_SIZE = 3


def _resize_limit(h, w, min_len=736, max_len=1280):
    """最短边>=736(放大), 最长边<=1280(缩小), 对齐32。"""
    if min(h, w) < min_len:
        ratio = min_len / min(h, w)
    else:
        ratio = 1.0
    rh, rw = int(h * ratio), int(w * ratio)
    if max(rh, rw) > max_len:
        s = max_len / max(rh, rw)
        rh, rw = int(rh * s), int(rw * s)
    rh = round(rh / 32) * 32
    rw = round(rw / 32) * 32
    return max(rh, 32), max(rw, 32)


def _select_shape(rh, rw):
    """选 H>=rh, W>=rw 的最小档位。"""
    for H, W in sorted(DET_SHAPES, key=lambda s: s[0] * s[1]):
        if H >= rh and W >= rw:
            return H, W
    return 1280, 1280


def preprocess(img):
    """返回 (input[1,3,H,W], meta=(rh,rw,H,W))。"""
    h, w = img.shape[:2]
    rh, rw = _resize_limit(h, w)
    H, W = _select_shape(rh, rw)
    resized = cv2.resize(img, (rw, rh))
    canvas = np.full((H, W, 3), 127.5, dtype=np.float32)
    canvas[:rh, :rw] = resized
    norm = (canvas / 255.0 - DET_MEAN) / DET_STD
    chw = norm.transpose(2, 0, 1)[np.newaxis]
    return chw, (rh, rw, H, W)


def _get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = np.array([points[index_1], points[index_2],
                    points[index_3], points[index_4]])
    return box, min(bounding_box[1])


def _box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] -= xmin
    box[:, 1] -= ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box, ratio):
    poly = Polygon(box)
    if poly.area < 1e-6:
        return box[np.newaxis]
    distance = poly.area * ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance)).reshape((-1, 1, 2))
    return expanded[:, 0]


def postprocess(pred, ori_shape, meta):
    """pred: [1,1,H,W]; meta=(rh,rw,H,W); 返回 boxes, 坐标映射回原图。"""
    src_h, src_w = ori_shape
    rh, rw, H, W = meta
    bitmap = pred[0, 0]
    segmentation = bitmap > DET_THRESH
    mask = segmentation.astype(np.uint8)
    if DET_USE_DILATION:
        mask = cv2.dilate(mask, np.array([[1, 1], [1, 1]]))
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours[:DET_MAX_CANDIDATES]:
        points, sside = _get_mini_boxes(contour)
        if sside < DET_MIN_SIZE:
            continue
        score = _box_score_fast(bitmap, points.reshape(-1, 2))
        if DET_BOX_THRESH > score:
            continue
        box = _unclip(points, DET_UNCLIP_RATIO)
        box, sside = _get_mini_boxes(box)
        if sside < DET_MIN_SIZE + 2:
            continue
        box = box.astype(np.float32)
        # 图像在 canvas 左上 [0:rh,0:rw], pad 在右下, 直接按比例映射回原图
        box[:, 0] = np.clip(np.round(box[:, 0] * src_w / rw), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] * src_h / rh), 0, src_h)
        boxes.append(box.astype(np.int32))
    return boxes
