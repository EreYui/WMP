# SPDX-License-Identifier: Apache-2.0
"""部署桥沿用参考环境的相机采样相位和一帧延迟。"""

import numpy as np


class ReferenceDepthBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.delayed = np.zeros((1, 64, 64, 1), dtype=np.float32)
        self.previous = self.delayed.copy()

    @staticmethod
    def capture_due(control_call):
        # reset 中先有一个零动作步；第4次策略动作之后对应环境全局第5步。
        return control_call % 5 == 3

    def capture(self, depth):
        self.delayed = self.previous.copy()
        self.previous = np.asarray(depth, dtype=np.float32).reshape(1, 64, 64, 1).copy()
