# SPDX-License-Identifier: Apache-2.0
"""实际感知模块分批推理的数值对照。"""

import torch
from wmp_lab.learning.inference import inference_chunks
from wmp_lab.learning.networks import DepthPredictor
from wmp_lab.learning.world_model import WorldModel


def test_perception_chunks_preserve_network_outputs():
    torch.manual_seed(29)
    prop = torch.randn(7, 33)
    heights = torch.randn(7, 525)
    predictor = DepthPredictor()
    world = WorldModel(feature_dim=64, stochastic_dim=8, hidden_dim=64, categories=8)
    with torch.no_grad():
        depth = predictor(heights, prop)
        chunked = inference_chunks(predictor, heights, prop, batch_size=2)
        torch.testing.assert_close(chunked, depth, rtol=2e-5, atol=2e-6)

        def encode(p, d):
            return world.encoder({"prop": p, "image": d})

        expected = encode(prop, depth)
        torch.testing.assert_close(inference_chunks(encode, prop, depth, batch_size=2), expected, rtol=2e-5, atol=2e-6)
