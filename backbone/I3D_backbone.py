# -*- coding: utf-8 -*-
# @Time: 2023/6/23 21:55
import os
import torch

from backbone.I3D import I3D
from utils.misc import fix_bn


def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class I3D_backbone(torch.nn.Module):
    def __init__(self, I3D_ckpt_path, I3D_class=400):
        super(I3D_backbone, self).__init__()

        self.backbone = I3D(I3D_class)
        self.load_pretrain(I3D_ckpt_path)
        

    def fixed_param(self):
        for param in self.backbone.parameters():  # Here we freez all the model's layers
            param.requires_grad = False

    def load_pretrain(self, I3D_ckpt_path):
        self.backbone.load_state_dict(torch.load(I3D_ckpt_path), strict=False)
        if int(os.environ.get("LOCAL_RANK", "-1")) < 1:
            print("Using I3D backbone:")
            print(f"\tLoad pretrained model from {I3D_ckpt_path}.")

    def get_feature_dim(self):
        return self.backbone.get_logits_dim()

    def forward(self, video, return_feats=False):
        batch_size, C, frames, H, W = video.shape

        # spatial-temporal feature
        if frames >= 160:
            start_idx = [i for i in range(0, frames, 16)]
        else:
            start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]

        clips = [video[:, :, i : i + 16] for i in start_idx]

        # Initialize a tensor to store clip-wise features
        clip_feats = {}
        clip_out = []

        # Extract features for each clip
        for i in range(len(start_idx)):
            feats, out = self.backbone(
                clips[i], return_feats=True
            )  # Extract shallow, mid, and deep features
            # merge dict
            if i == 0:
                for k in feats.keys():
                    clip_feats[k] = [feats[k].unsqueeze(-1)]
            else:
                for k in feats.keys():
                    clip_feats[k].append(feats[k].unsqueeze(-1))
            clip_out.append(out)

        # clip-wise mean pooling
        clip_out = torch.cat(clip_out, dim=-1)
        clip_out = clip_out.mean(-1).view(batch_size, -1)
        for k in clip_feats.keys():
            clip_feats[k] = torch.cat(clip_feats[k], dim=-1).mean(-1).mean(-1).mean(-1)
            # B, C, T, H, W -> B, C
            clip_feats[k] = clip_feats[k].view(batch_size, -1)

        if return_feats:
            return clip_feats, clip_out
        else:
            return clip_out  # or any other feature, depending on which one you need to return


# Example usage
if __name__ == "__main__":
    # Create a sample input tensor with shape (batch_size, channels, frames, height, width)
    batch_size = 1
    channels = 3  # For RGB
    frames = 103  # Frame count
    height = 224
    width = 224
    sample_input = torch.randn(batch_size, channels, frames, height, width)

    # Initialize the I3D model
    num_classes = 400  # Example number of classes
    model = I3D_backbone(I3D_ckpt_path="weights/model_rgb.pth", I3D_class=num_classes)

    # Forward pass
    shallow_feats, mid_feats, deep_feats = model(sample_input, return_feats=True)

    print(f"Shallow feature shape: {shallow_feats.shape}")
    print(f"Mid feature shape: {mid_feats.shape}")
    print(f"Deep feature shape: {deep_feats.shape}")
