# SPDX-License-Identifier: Apache-2.0
"""按环境分批推理，限制无梯度感知网络的临时显存。"""

import torch


def inference_chunks(function, *inputs, batch_size=512):
    if inputs[0].shape[0] <= batch_size:
        return function(*inputs)
    return torch.cat(
        [
            function(*(value[start : start + batch_size] for value in inputs))
            for start in range(0, inputs[0].shape[0], batch_size)
        ],
        dim=0,
    )
