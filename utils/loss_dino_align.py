#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, importlib.util, math
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ TorchHub: 安全加载 DINO ------------------------
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


# ------------------------ 工具：线性插值矩阵（初始化） ------------------------
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


# ------------------------ 轴向 MLP（用于空间轴） ------------------------
class AxisMLPAligner(nn.Module):
    """
    在某一轴长度 L_in 上对齐到 L_out：
      y = Interp(L_in→L_out)(x) + α * MLP(x)   （α 初始 0）
    MLP: 两层 GELU，隐藏维 = expand * L_in
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

        self.base = nn.Linear(L_in, L_out, bias=False)
        with torch.no_grad():
            W = (
                _linear_resample_matrix(
                    L_in, L_out, self.base.weight.device, self.base.weight.dtype
                )
                if init_mode == "interp"
                else torch.eye(
                    L_out,
                    L_in,
                    device=self.base.weight.device,
                    dtype=self.base.weight.dtype,
                )
            )
            self.base.weight.copy_(W)

        self.mlp = nn.Sequential(
            nn.Linear(L_in, hid, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, L_out, bias=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # 残差门控（初始0→等价插值）

    def forward(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        x = x.movedim(axis, -1)  # [..., L_in]
        shp = x.shape[:-1]
        x2d = x.reshape(-1, self.L_in)  # [N, L_in]
        y = self.base(x2d) + self.alpha * self.mlp(x2d)
        y = y.view(*shp, self.L_out).movedim(-1, axis)
        return y


# ------------------------ 时间轴：线性插值（可固定/可学习） ------------------------
class TimeLinearAligner(nn.Module):
    """
    时间轴对齐：插值等效的 Linear(T -> T_target)
      - learnable=False: 固定插值
      - learnable=True : 可学习插值（初始化=插值）
    输入 x: [B,T,H,W] -> 输出: [B,T_target,H,W]
    """

    def __init__(
        self,
        T: int,
        T_target: int,
        learnable: bool = True,
        init_mode: Literal["interp", "identity"] = "interp",
    ):
        super().__init__()
        self.proj = nn.Linear(T, T_target, bias=False)
        with torch.no_grad():
            W = (
                _linear_resample_matrix(
                    T, T_target, self.proj.weight.device, self.proj.weight.dtype
                )
                if init_mode == "interp"
                else torch.eye(
                    T_target,
                    T,
                    device=self.proj.weight.device,
                    dtype=self.proj.weight.dtype,
                )
            )
            self.proj.weight.copy_(W)
        if not learnable:
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bthw,ft->bfhw", x, self.proj.weight)


# ------------------------ 独立的时空对齐模块（缩放网络） ------------------------
class TSAlignerHybrid(nn.Module):
    """
    输入:  x[B,Tf,H',W']  →  输出: y[B,T,H,W]
      - 时间轴: TimeLinearAligner（'interp_fixed' or 'interp_learnable'）或 AxisMLPAligner('mlp')
      - 空间轴: AxisMLPAligner（H、W 各一套，插值初始化+残差 MLP）
    """

    def __init__(
        self,
        Tf: int,
        H_in: int,
        W_in: int,
        T: int,
        H: int,
        W: int,
        time_mode: Literal["interp_fixed", "interp_learnable", "mlp"] = "interp_fixed",
        expand_space: float = 1.0,
        dropout: float = 0.0,
        init_mode: Literal["interp", "identity"] = "interp",
    ):
        super().__init__()
        if time_mode == "mlp":
            self.time = AxisMLPAligner(
                Tf, T, expand=1.0, dropout=dropout, init_mode=init_mode
            )
        else:
            self.time = TimeLinearAligner(
                Tf, T, learnable=(time_mode == "interp_learnable"), init_mode=init_mode
            )
        self.h = AxisMLPAligner(
            H_in, H, expand=expand_space, dropout=dropout, init_mode=init_mode
        )
        self.w = AxisMLPAligner(
            W_in, W, expand=expand_space, dropout=dropout, init_mode=init_mode
        )

        self.src = (Tf, H_in, W_in)
        self.tgt = (T, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,Tf,H',W']  ->  y: [B,T,H,W]
        """
        if x.dim() != 4:
            raise ValueError(
                f"TSAlignerHybrid expects [B,Tf,H',W'], got {list(x.shape)}"
            )
        B, Tf, H_in, W_in = x.shape
        Tf0, H0, W0 = self.src
        if (Tf, H_in, W_in) != (Tf0, H0, W0):
            raise ValueError(
                f"Aligner was built for src={self.src}, but got {(Tf,H_in,W_in)}"
            )
        y = self.time(x)  # [B,T,H',W']
        y = self.h(y, axis=2)  # [B,T,H,W']
        y = self.w(y, axis=3)  # [B,T,H,W]
        return y


# ------------------------ DINO 教师：从已归一化视频生成热图 ------------------------
class DinoTeacher(nn.Module):
    """
    从（已归一化的）视频帧生成 DINO 注意力热图：
      video: [B,T,3,H,W] or [B,3,T,H,W]  →  heatmaps: [B,T,H,W]
    """

    def __init__(
        self,
        dino_name: str = "dino_vits16",
        dino_in_size: int = 224,
        assume_pre_normalized: bool = True,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.teacher = safe_load_dino(dino_name)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.dino_in_size = int(dino_in_size)
        self.assume_pre_normalized = assume_pre_normalized
        self.chunk_size = chunk_size

        # (可选) ImageNet 归一化（仅在 assume_pre_normalized=False 时使用）
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def _maybe_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.assume_pre_normalized:
            return x
        if x.max() > 1.5:  # 支持 0~255
            x = x / 255.0
        mean = self._mean.to(x.device, x.dtype)
        std = self._std.to(x.device, x.dtype)
        return (x - mean) / std

    @torch.no_grad()
    def _per_frame_attn(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [N,3,H,W]  ->  heatmaps: [N,H,W]
        """
        N, _, H, W = frames.shape
        x = F.interpolate(
            frames,
            size=(self.dino_in_size, self.dino_in_size),
            mode="bilinear",
            align_corners=False,
        )
        x = self._maybe_norm(x)
        attn = self.teacher.get_last_selfattention(x)  # [N, heads, Ntok, Ntok]
        cls2patch = attn[:, :, 0, 1:].mean(dim=1)  # [N, Np]
        gh = gw = int(math.sqrt(cls2patch.size(1)))
        hm = cls2patch.view(N, 1, gh, gw)
        hm = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)
        return hm.squeeze(1)  # [N,H,W]

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B,T,3,H,W] or [B,3,T,H,W]  ->  [B,T,H,W]
        """
        if video.dim() != 5:
            raise ValueError("video must be 5D")
        if video.shape[1] == 3:
            video = video.transpose(1, 2)  # [B,3,T,H,W] -> [B,T,3,H,W]
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        if self.chunk_size is None:
            hm = self._per_frame_attn(frames)  # [B*T,H,W]
        else:
            chunks = []
            for i in range(0, B * T, self.chunk_size):
                chunks.append(self._per_frame_attn(frames[i : i + self.chunk_size]))
            hm = torch.cat(chunks, dim=0)
        return hm.view(B, T, H, W)


# ------------------------ DINO JSD Loss：接收“对齐后的热图” + “已归一化视频” ------------------------
class DinoJSDLoss(nn.Module):
    """
    比较：
      - student_aligned: 对齐后的学生热图 [B,T,H,W]（由外部 TSAlignerHybrid 生成）
      - video_norm:      已归一化视频 [B,T,3,H,W] 或 [B,3,T,H,W]
    内部：调用 DinoTeacher 生成 teacher_heatmap（同形状），计算 JSD（温度 softmax）+ 可选时间平滑。
    注意：本模块**不做对齐**，若形状不一致会报错提示用对齐网络。
    """

    def __init__(
        self,
        dino_name: str = "dino_vits16",
        dino_in_size: int = 224,
        temperature: float = 1.0,
        lambda_temporal: float = 0.0,
        assume_pre_normalized: bool = True,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_temporal = float(lambda_temporal)
        self.teacher = DinoTeacher(
            dino_name=dino_name,
            dino_in_size=dino_in_size,
            assume_pre_normalized=assume_pre_normalized,
            chunk_size=chunk_size,
        )

    # 过滤掉 teacher 参数：crit.parameters() 不含 teacher.*
    def named_parameters(self, *args, **kwargs):
        for name, p in super().named_parameters(*args, **kwargs):
            if name.startswith("teacher."):
                continue
            yield name, p

    def parameters(self, recurse: bool = True):
        for _, p in self.named_parameters(recurse=recurse):
            yield p

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

    def forward(
        self, student_aligned: torch.Tensor, video_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        student_aligned: [B,T,H,W]  —— 外部对齐网络输出
        video_norm:      [B,T,3,H,W] 或 [B,3,T,H,W] —— 已做归一化
        """
        if student_aligned.dim() != 4:
            raise ValueError(
                f"student_aligned must be [B,T,H,W], got {list(student_aligned.shape)}"
            )

        B, T, H, W = student_aligned.shape
        with torch.no_grad():
            teacher_hm = self.teacher(video_norm)  # [B,T0,H0,W0]

        if teacher_hm.shape != (B, T, H, W):
            raise ValueError(
                f"teacher heatmap shape {list(teacher_hm.shape)} != student_aligned {list(student_aligned.shape)}. "
                f"请先用 TSAlignerHybrid 对齐到一致的 [B,T,H,W]。"
            )

        # 展平为概率，逐帧 JSD
        p = self._to_prob(student_aligned.reshape(B * T, H, W), self.temperature)
        q = self._to_prob(teacher_hm.reshape(B * T, H, W), self.temperature)
        loss_jsd = self._jsd(p, q)

        # 可选：学生概率时间平滑
        if self.lambda_temporal > 0 and T > 1:
            p_bt_hw = p.view(B, T, -1)
            tv = (p_bt_hw[:, 1:] - p_bt_hw[:, :-1]).abs().mean()
            return loss_jsd + self.lambda_temporal * tv
        else:
            return loss_jsd


# ------------------------ 最小用例 ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    # 假设我们有一段已归一化的视频（16帧，224x224）
    B, T, H, W = 1, 16, 224, 224
    video_norm = torch.randn(B, T, 3, H, W, device=device)  # 已归一化

    # 学生产生的热图（分辨率更低、时间更短）：Tf=8, Hp=56, Wp=56
    Tf, Hp, Wp = 8, H // 4, W // 4
    student_heatmap_raw = torch.randn(B, Tf, Hp, Wp, device=device)

    # 1) 建对齐模块：把 [B,Tf,Hp,Wp] → [B,T,H,W]
    aligner = TSAlignerHybrid(
        Tf=Tf,
        H_in=Hp,
        W_in=Wp,
        T=T,
        H=H,
        W=W,
        time_mode="interp_fixed",  # 时间用固定插值；或 "interp_learnable" / "mlp"
        expand_space=2.0,  # 空间 MLP 表达力
        dropout=0.0,
        init_mode="interp",
    ).to(device)

    student_aligned = aligner(student_heatmap_raw)  # [B,T,H,W]
    print("Aligned student heatmap shape:", student_aligned.shape)

    # 2) DINO JSD Loss：接收“对齐后的热图” + “已归一化视频”
    crit = DinoJSDLoss(
        dino_name="dino_vits16",
        dino_in_size=224,
        temperature=1.0,
        lambda_temporal=0.05,
        assume_pre_normalized=True,  # 你的视频已归一化
        chunk_size=8,
    ).to(device)

    with torch.no_grad():
        loss = crit(student_aligned, video_norm)
        print("DINO JSD loss:", float(loss))
