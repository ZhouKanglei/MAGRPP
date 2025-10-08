# -*- coding: utf-8 -*-
# @Time: 2023/6/22 23:01
import torch

from pydoc import locate
from backbone import Backbone, xavier

from backbone.TSAlignerHybrid import TSAlignerHybrid


class AQAMLP(Backbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, args):
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(AQAMLP, self).__init__()

        feature_extractor = locate(args.feature_extractor)
        self.feature_extractor = feature_extractor(**args.feature_extractor_args)

        regressor = locate(args.regressor)
        self.regressor = regressor(**args.regressor_args)

        if args.projector is not None:
            projector = locate(args.projector)
            self.projector = projector(**args.projector_args)

        if args.model == "magrp":
            dummy_input = torch.randn(1, 3, 103, 224, 224).float()
            feats, _, _ = self.feature_extractor(dummy_input, return_feats=True)
            Hp, Wp, Tf = feats.shape[-3], feats.shape[-2], feats.shape[2]
            T, H, W = 16, 224, 224

            self.ts_aligner = TSAlignerHybrid(
                Tf=Tf,
                H_in=Hp,
                W_in=Wp,
                T=T,
                H=H,
                W=W,
                time_mode="mlp",
                expand_space=5.0,
                dropout=0.0,
                init_mode="interp",
            )

        self.args = args

    def reset_parameters(self):
        """
        Calls the Xavier parameter initialization function.
        """
        self.regressor.apply(xavier)

    def forward_phaseA(self, x):
        """Only forward pass for phase A."""
        feats = self.feature_extractor(x)

        return feats

    def forward_phaseB(self, buf_feats, old_feats):
        """ " Forward pass for phase B, which includes projector."""
        buf_hat = self.projector(buf_feats)
        old_hat = self.projector(old_feats)

        return buf_hat, old_hat

    def forward_phaseC(self, feats, buf_hat_feats):
        """Forward pass for phase C, which includes regressor."""
        out = self.regressor(feats)
        buf_out = self.regressor(buf_hat_feats)

        return out, buf_out

    def forward(
        self,
        x,
        buf_feats=None,
        old_feats=None,
        returnt="out",
        return_feats=False,
        replay=False,
    ):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        if return_feats:
            mid_feats, deep_feats = self.feature_extractor(
                x, return_feats=return_feats
            )
            feats = deep_feats
        else:
            feats = self.feature_extractor(x)
        if hasattr(self, "projector") and self.args.projector is not None:
            zero = torch.zeros((), device=feats.device, dtype=feats.dtype)
            feats = feats + self.projector(feats) * zero
        out = self.regressor(feats)

        reply_out = {}
        if buf_feats is not None:
            reply_out["buf_hat"] = self.projector(buf_feats)
            reply_out["buf_out"] = self.regressor(reply_out["buf_hat"])

        if old_feats is not None:
            reply_out["old_hat"] = self.projector(old_feats)
            reply_out["old_out"] = self.regressor(reply_out["old_hat"])

        if buf_feats is None and old_feats is None:
            if returnt == "out":
                return out
            elif returnt == "all" and return_feats:
                return out, (mid_feats, feats)
            elif returnt == "all" and not return_feats:
                return out, feats
        else:
            if returnt == "out":
                return out, reply_out
            elif returnt == "all" and not return_feats:
                return out, feats, reply_out
            elif returnt == "all_feats" and return_feats:
                return out, (mid_feats, feats), reply_out
