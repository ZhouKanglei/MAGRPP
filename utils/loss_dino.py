#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, importlib.util, math
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ Safe DINO loader ------------------------
def _inject_dino_utils_if_needed():
    hub_dir = torch.hub.get_dir()
    dino_dir = os.path.join(hub_dir, "facebookresearch_dino_main")
    utils_py = os.path.join(dino_dir, "utils.py")
    if not os.path.exists(utils_py):
        return False
    spec = importlib.util.spec_from_file_location("utils", utils_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    sys.modules["utils"] = mod
    return True


def safe_load_dino(dino_name="dino_vits16", **kwargs):
    try:
        return torch.hub.load("facebookresearch/dino:main", dino_name, **kwargs)
    except ImportError as e:
        if "from 'utils'" in str(e) or "trunc_normal_" in str(e):
            if _inject_dino_utils_if_needed():
                return torch.hub.load("facebookresearch/dino:main", dino_name, **kwargs)
        raise


# ------------------------ 可选：ImageNet 归一化 ------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _imgnet_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.max() > 1.5:  # 支持 0~255
        x = x / 255.0
    mean = x.new_tensor(_IMAGENET_MEAN)[None, :, None, None]
    std = x.new_tensor(_IMAGENET_STD)[None, :, None, None]
    return (x - mean) / std


# ------------------------ 线性插值矩阵（用于初始化） ------------------------
def _linear_resample_matrix(in_len: int, out_len: int, device, dtype):
    if in_len == out_len:
        return torch.eye(out_len, in_len, device=device, dtype=dtype)
    src = torch.linspace(0, in_len - 1, out_len, device=device, dtype=dtype)
    x0 = torch.floor(src).clamp(0, in_len - 1)
    x1 = (x0 + 1).clamp(0, in_len - 1)
    w1 = src - x0
    w0 = 1.0 - w1
    W = torch.zeros(out_len, in_len, device=device, dtype=dtype)
    idx = torch.arange(out_len, device=device)
    x0 = x0.long()
    x1 = x1.long()
    W[idx, x0] += w0
    W[idx, x1] += w1
    return W


# ------------------------ 轴向 MLP 对齐器 ------------------------
class AxisMLPAligner(nn.Module):
    """
    在某一轴长为 L_in 的维度上，把长度对齐到 L_out：
      y = Interp(L_in→L_out)(x) + α * MLP(x)   （α 初始 0，稳定）
    MLP: 两层，gelu，隐藏维=expand*L_in（可调）；Dropout 可选。
    输入张量会被重排，使被对齐维在最后一维，然后按 [*, L_in] → [*, L_out] 处理。
    """

    def __init__(
        self,
        L_in: int,
        L_out: int,
        expand: float = 1.0,
        dropout: float = 0.0,
        init_mode: Literal["interp", "identity"] = "interp",
    ):
        super().__init__()
        hid = max(1, int(round(expand * L_in)))
        self.L_in, self.L_out = L_in, L_out

        # 基线：线性插值等效的 Linear
        self.base = nn.Linear(L_in, L_out, bias=False)
        with torch.no_grad():
            self.base.weight.copy_(
                _linear_resample_matrix(
                    L_in,
                    L_out,
                    device=self.base.weight.device,
                    dtype=self.base.weight.dtype,
                )
                if init_mode == "interp"
                else torch.eye(
                    L_out,
                    L_in,
                    device=self.base.weight.device,
                    dtype=self.base.weight.dtype,
                )
            )

        # 残差 MLP（两层）
        self.mlp = nn.Sequential(
            nn.Linear(L_in, hid, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, L_out, bias=True),
        )
        # 门控系数 α，初始 0 → 一开始完全等价于插值
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        """
        x: 任意形状张量，沿 axis 做对齐
        """
        # 把 axis 放到最后一维
        x = x.movedim(axis, -1)  # [..., L_in]
        shape_flat = x.shape[:-1]
        x2d = x.reshape(-1, self.L_in)  # [N, L_in]

        y = self.base(x2d) + self.alpha * self.mlp(x2d)  # [N, L_out]
        y = y.view(*shape_flat, self.L_out)

        # 再把维度移回原位（但长度已变为 L_out）
        y = y.movedim(-1, axis)  # [... (axis=L_out) ...]
        return y


class TSMLPAligner(nn.Module):
    """
    时空分离 MLP 对齐器：
      输入:  x[B, T, H, W]
      输出:  y[B, Tf, Hp, Wp]
    在 T 轴、H 轴、W 轴各用一个 AxisMLPAligner（带插值初始化 + 残差 MLP）。
    """

    def __init__(
        self,
        T,
        H,
        W,
        Tf,
        Hp,
        Wp,
        expand_time: float = 1.0,
        expand_space: float = 1.0,
        dropout: float = 0.0,
        init_mode: str = "interp",
    ):
        super().__init__()
        self.T_src, self.H_src, self.W_src = T, H, W
        self.T_tgt, self.H_tgt, self.W_tgt = Tf, Hp, Wp
        self.time = AxisMLPAligner(
            T, Tf, expand=expand_time, dropout=dropout, init_mode=init_mode
        )
        self.h = AxisMLPAligner(
            H, Hp, expand=expand_space, dropout=dropout, init_mode=init_mode
        )
        self.w = AxisMLPAligner(
            W, Wp, expand=expand_space, dropout=dropout, init_mode=init_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,H,W] → 时间对齐
        y = self.time(x, axis=1)  # [B,Tf,H,W]
        # 高度
        y = self.h(y, axis=2)  # [B,Tf,Hp,W]
        # 宽度
        y = self.w(y, axis=3)  # [B,Tf,Hp,Wp]
        return y


# ------------------------ 主类：视频 DINO 监督（JSD + 轴向 MLP 对齐） ------------------------
class VideoDinoHeatmapLossFlex(nn.Module):
    """
    用 DINO 最后一层 CLS→patch 注意力热图监督视频中间层。
    - 对齐：TSMLPAligner（时间/空间轴向 MLP，对齐器自带插值初始化+可学习残差）
    - 学生热图：通道聚合 ('l2' 或 'abs_mean')
    - 损失：温度 softmax 后做 JSD，可选时间平滑 TV
    """

    def __init__(
        self,
        dino_name: str = "dino_vits16",
        dino_in_size: int = 224,
        student_reduce: Literal["l2", "abs_mean"] = "l2",
        temperature: float = 1.0,
        lambda_temporal: float = 0.0,
        chunk_size: Optional[int] = None,
        # 对齐器参数
        init_mode: Literal["interp", "identity"] = "interp",
        expand_time: float = 1.0,
        expand_space: float = 1.0,
        dropout: float = 0.0,
        freeze_aligner: bool = False,
        assume_pre_normalized: bool = True,
    ):
        super().__init__()
        assert student_reduce in {"l2", "abs_mean"}
        self.dino_in_size = int(dino_in_size)
        self.student_reduce = student_reduce
        self.temperature = float(temperature)
        self.lambda_temporal = float(lambda_temporal)
        self.chunk_size = chunk_size
        self.init_mode = init_mode
        self.expand_time = float(expand_time)
        self.expand_space = float(expand_space)
        self.dropout = float(dropout)
        self.freeze_aligner = freeze_aligner
        self.assume_pre_normalized = assume_pre_normalized

        # DINO 教师（冻结）
        self.teacher = safe_load_dino(dino_name)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # 对齐器懒初始化
        self.aligner: Optional[TSMLPAligner] = None

    # ---- 教师：单帧注意力热图 ----
    @torch.no_grad()
    def _dino_attn_per_frame(self, frames: torch.Tensor) -> torch.Tensor:
        N, _, H, W = frames.shape
        x = F.interpolate(
            frames,
            size=(self.dino_in_size, self.dino_in_size),
            mode="bilinear",
            align_corners=False,
        )
        if not self.assume_pre_normalized:
            x = _imgnet_norm(x)
        attn = self.teacher.get_last_selfattention(x)  # [N, heads, Ntok, Ntok]
        cls2patch = attn[:, :, 0, 1:].mean(dim=1)  # [N, Np]
        gh = gw = int(math.sqrt(cls2patch.size(1)))
        hm = cls2patch.view(N, 1, gh, gw)
        hm = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)
        return hm.squeeze(1)  # [N,H,W]

    @torch.no_grad()
    def _teacher_video_heatmaps(self, video_bt3hw: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is None:
            return self._dino_attn_per_frame(video_bt3hw)
        outs = []
        for i in range(0, video_bt3hw.size(0), self.chunk_size):
            outs.append(self._dino_attn_per_frame(video_bt3hw[i : i + self.chunk_size]))
        return torch.cat(outs, dim=0)

    # ---- 学生热图 ----
    def _student_map(self, feat_bt_c_hw: torch.Tensor) -> torch.Tensor:
        f = feat_bt_c_hw.float()
        if self.student_reduce == "l2":
            return torch.sqrt(torch.clamp((f**2).sum(dim=1), min=1e-12))
        else:
            return f.abs().mean(dim=1)

    # ---- 概率 & JSD ----
    @staticmethod
    def _to_prob(vmap_bt_hw: torch.Tensor, temperature: float) -> torch.Tensor:
        v = vmap_bt_hw.view(vmap_bt_hw.size(0), -1)
        if temperature != 1.0:
            v = v / temperature
        return F.softmax(v, dim=1)

    @staticmethod
    def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        m = 0.5 * (p + q)
        return 0.5 * (
            F.kl_div((p + eps).log(), m, reduction="batchmean")
            + F.kl_div((q + eps).log(), m, reduction="batchmean")
        )

    def _ensure_aligner(self, T, H, W, Tf, Hp, Wp, device, dtype):
        if self.aligner is None or (
            self.aligner.T_src,
            self.aligner.H_src,
            self.aligner.W_src,
            self.aligner.T_tgt,
            self.aligner.H_tgt,
            self.aligner.W_tgt,
        ) != (T, H, W, Tf, Hp, Wp):
            self.aligner = TSMLPAligner(
                T,
                H,
                W,
                Tf,
                Hp,
                Wp,
                expand_time=self.expand_time,
                expand_space=self.expand_space,
                dropout=self.dropout,
                init_mode=self.init_mode,
            ).to(device=device, dtype=dtype)
            if self.freeze_aligner:
                for p in self.aligner.parameters():
                    p.requires_grad = False

    # --- 只导出“非 teacher”的可训练参数 ---
    def named_parameters(self, *args, **kwargs):
        """过滤掉 teacher.* 的参数"""
        for name, p in super().named_parameters(*args, **kwargs):
            # 顶层名以 'teacher.' 开头的全部跳过
            if name.startswith("teacher."):
                continue
            yield name, p

    def parameters(self, recurse: bool = True):
        """使 crit.parameters() 与上面的过滤保持一致"""
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    # ---- 前向 ----
    def forward(
        self,
        video: torch.Tensor,  # [B,T,3,H,W] or [B,3,T,H,W]（已归一化建议 assume_pre_normalized=True）
        feat: torch.Tensor,  # [B,Tf,C,H',W'] / [B,C,Tf,H',W'] / [B,C,H',W']
    ) -> torch.Tensor:

        # 统一 video → [B,T,3,H,W]
        if video.dim() != 5:
            raise ValueError("video must be 5D")
        if video.shape[1] == 3:
            video = video.transpose(1, 2)
        B, T, _, H, W = video.shape

        # 统一 feat → [B,Tf,C,H',W']，支持时间池化输入 [B,C,H',W']
        if feat.dim() == 4:
            feat = feat.unsqueeze(1)  # Tf=1
        elif (
            feat.dim() == 5
            and feat.shape[1] != video.shape[1]
            and feat.shape[2] == video.shape[1]
        ):
            feat = feat.transpose(1, 2)
        elif feat.dim() != 5:
            raise ValueError("feat must be 4D or 5D")
        Bf, Tf, C, Hp, Wp = feat.shape
        assert Bf == B, "batch size mismatch"

        # 教师热图（逐帧）
        video_bt3hw = video.reshape(B * T, 3, H, W)
        with torch.no_grad():
            teacher_bt_hw = self._teacher_video_heatmaps(video_bt3hw)  # [B*T,H,W]
            teacher_b_t_hw = teacher_bt_hw.view(B, T, H, W)  # [B,T,H,W]

        # 懒创建/更新 MLP 对齐器；把教师热图对齐成 [B,Tf,Hp,Wp]
        self._ensure_aligner(
            T, H, W, Tf, Hp, Wp, device=video.device, dtype=teacher_b_t_hw.dtype
        )
        aligned_teacher = self.aligner(teacher_b_t_hw)  # [B,Tf,Hp,Wp]

        # 学生热图 [B,Tf,C,Hp,Wp] → [B,Tf,Hp,Wp]
        student_bt_c_hw = feat.reshape(B * Tf, C, Hp, Wp)
        student_b_t_hw = self._student_map(student_bt_c_hw).view(B, Tf, Hp, Wp)

        # JSD（逐帧）
        p = self._to_prob(student_b_t_hw.reshape(B * Tf, Hp, Wp), self.temperature)
        q = self._to_prob(aligned_teacher.reshape(B * Tf, Hp, Wp), self.temperature)
        loss_jsd = self._jsd(p, q)

        # 可选：学生时间平滑
        if self.lambda_temporal > 0 and Tf > 1:
            p_bt_hw = p.view(B, Tf, -1)
            tv = (p_bt_hw[:, 1:] - p_bt_hw[:, :-1]).abs().mean()
            return loss_jsd + self.lambda_temporal * tv
        else:
            return loss_jsd


# ------------------------ 最小自测 ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    B, T, H, W = 1, 16, 224, 224
    video = torch.randn(B, T, 3, H, W, device=device)  # 假设已归一化
    feat_Tf16 = torch.randn(B, 5, 221, H // 2, W // 2, device=device)
    feat_Tf1 = torch.randn(B, 128, H // 4, W // 4, device=device)
    feat_Tf8 = torch.randn(B, 8, 128, H // 4, W // 4, device=device)

    crit = VideoDinoHeatmapLossFlex(
        dino_name="dino_vits16",
        dino_in_size=224,
        student_reduce="l2",
        temperature=1.0,
        lambda_temporal=0.05,
        chunk_size=8,
        init_mode="interp",  # MLP 基线=插值
        expand_time=0.25,  # 把时间/空间 MLP 的隐藏扩展倍率调大可增强表达力，如 2.0
        expand_space=0.25,
        dropout=0.0,
        freeze_aligner=False,
        assume_pre_normalized=True,
    ).to(device)

    with torch.no_grad():
        print("loss Tf=16:", float(crit(video, feat_Tf16)))
        print("loss Tf=1 :", float(crit(video, feat_Tf1)))
        print("loss Tf=8 :", float(crit(video, feat_Tf8)))

    # model trainable params
    n_params = sum(p.numel() for p in crit.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e3:.1f}K")
