"""RKNNLite 推理引擎封装(板端运行)。"""
import numpy as np
from rknnlite.api import RKNNLite


class RKNNEngine:
    """单个 RKNN 模型的加载与推理封装。"""

    def __init__(self, model_path: str, core_mask=RKNNLite.NPU_CORE_0_1):
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn 失败(ret={ret}): {model_path}")
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime 失败(ret={ret}): {model_path}")

    def infer(self, inputs):
        """inputs: np.ndarray 或 list[np.ndarray] -> list[np.ndarray]。
        输入是 float32 NCHW(已归一化), 显式指定 data_type/data_format。"""
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        return self.rknn.inference(
            inputs=inputs, data_type="float32", data_format="nchw"
        )

    def release(self):
        self.rknn.release()
