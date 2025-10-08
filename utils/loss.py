# -*- coding: utf-8 -*-
# @Time: 2023/10/5 19:54
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_js_loss(feats_1, feats_2, epsilon=1e-8):
    """
    Compute the Jensen-Shannon Divergence loss between two feature maps, averaging over channels.

    Args:
    - epsilon: Small constant to avoid division by zero when computing JS divergence

    Returns:
    - JS loss: Scalar value representing the JS divergence between the feature maps
    """

    # Flatten the feature maps to [batch_size, height * width]
    batch_size = feats_1.shape[0]
    feats_1 = feats_1.view(batch_size, -1)  # Shape: [batch_size, height * width]
    feats_2 = feats_2.view(batch_size, -1)  # Shape: [batch_size, height * width]

    # Apply softmax to normalize the feature maps into probability distributions
    probs_1 = F.softmax(feats_1, dim=-1)  # Shape: [batch_size, height * width]
    probs_2 = F.softmax(feats_2, dim=-1)  # Shape: [batch_size, height * width]

    # Compute the mean distribution M = 0.5 * (P + Q)
    mean_probs = 0.5 * (probs_1 + probs_2)

    # Compute KL divergence for P || M and Q || M
    kl_1 = F.kl_div(probs_1.log(), mean_probs, reduction="batchmean")
    kl_2 = F.kl_div(probs_2.log(), mean_probs, reduction="batchmean")

    # Compute JS divergence: JS(P || Q) = 0.5 * (KL(P || M) + KL(Q || M))
    js_loss = 0.5 * (kl_1 + kl_2)

    return kl_1


def pairwise_distance_loss(feats, s, p=2, normalize=True):
    """
    让 feats 的距离关系与 s 的距离关系一致的 loss。

    feats: [B, D] 特征张量
    s:     [B]    质量分数
    p:     距离范数（默认2，即欧式距离）
    normalize: 是否对距离矩阵归一化到 [0,1]，防止尺度影响
    """
    # 计算 feats 的两两距离
    D_f = torch.cdist(feats, feats, p=p)  # [B, B]
    D_f = D_f.to(torch.float32)

    # 计算 s 的两两差值
    if s.shape[1] == 2:
        s = s[:, :1]
    D_s = torch.cdist(s, s, p=1)  # [B, B]
    D_s = D_s.to(torch.float32)

    if normalize:
        D_f = D_f / (D_f.max() + 1e-8)
        D_s = D_s / (D_s.max() + 1e-8)

    # L2 损失对齐两个距离矩阵
    loss = F.mse_loss(D_f, D_s)
    return loss


class AngularJSDAlignLoss(nn.Module):
    def __init__(
        self,
        angle_mode: str = "cosdist",
        temperature: float = 0.2,
        use_row_minmax: bool = True,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        assert angle_mode in ("acos", "cosdist")
        assert reduction in ("mean", "sum")
        self.angle_mode = angle_mode
        self.tau = float(temperature)
        self.use_row_minmax = use_row_minmax
        self.reduction = reduction
        self.eps = eps

    def _row_minmax(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        xmin = x.min(dim=1, keepdim=True).values
        xmax = x.max(dim=1, keepdim=True).values
        return (x - xmin) / (xmax - xmin + self.eps)

    def _masked_row_softmax(
        self, scores: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=1)

    def _angular_distance(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.float()
        f = F.normalize(feats, dim=-1)
        cos_ij = (f @ f.t()).clamp(-1 + 1e-6, 1 - 1e-6)
        if self.angle_mode == "acos":
            return torch.acos(cos_ij)
        else:
            return 1.0 - cos_ij

    @torch.no_grad()
    def _score_diff(self, s: torch.Tensor) -> torch.Tensor:
        return torch.cdist(s.unsqueeze(1).float(), s.unsqueeze(1).float(), p=1)

    def _row_probs_from_dist(self, D: torch.Tensor) -> torch.Tensor:
        D = D.float()
        R, C = D.shape
        # 行归一化提升数值稳定
        if self.use_row_minmax:
            xmin = D.min(dim=1, keepdim=True).values
            xmax = D.max(dim=1, keepdim=True).values
            Dn = (D - xmin) / (xmax - xmin + self.eps)
        else:
            Dn = D

        logits = -Dn / self.tau

        # 构造与 D 同形状的遮罩
        if R == C:
            mask = ~torch.eye(R, dtype=torch.bool, device=D.device)  # 方阵：去掉对角
        else:
            mask = torch.ones(
                (R, C), dtype=torch.bool, device=D.device
            )  # 非方阵：不过滤

        return self._masked_row_softmax(logits, mask)

    def _jsd_rowwise(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p, q = p.float(), q.float()
        m = 0.5 * (p + q)
        log_p = (p + self.eps).log()
        log_q = (q + self.eps).log()
        log_m = (m + self.eps).log()
        kl_pm = F.kl_div(log_p, m, reduction="batchmean")
        kl_qm = F.kl_div(log_q, m, reduction="batchmean")
        return 0.5 * (kl_pm + kl_qm)

    def _block_loss(self, Df: torch.Tensor, Ds: torch.Tensor) -> torch.Tensor:
        # If the block is too small (no comparable elements), return 0 directly
        Rf, Cf = Df.shape
        Rs, Cs = Ds.shape
        if min(Rf, Cf, Rs, Cs) <= 1:
            return Df.new_tensor(0.0)

        pf = self._row_probs_from_dist(Df)
        ps = self._row_probs_from_dist(Ds)
        return self._jsd_rowwise(pf, ps)

    def forward(self, feats, s, blocking=None):
        feats = feats.float()
        s = s.float()

        if len(feats.shape) == 3:
            feats = feats.mean(-1)
        if s.ndim == 2 and s.shape[1] == 2:
            s = s[:, :1]
        s = s.squeeze(1)

        B = feats.size(0)
        Df = self._angular_distance(feats)
        Ds = self._score_diff(s)

        losses = [self._block_loss(Df, Ds)]

        if blocking is not None:
            b1 = int(blocking)
            assert 0 < b1 < B
            losses.append(self._block_loss(Df[:b1, :b1], Ds[:b1, :b1]))
            losses.append(self._block_loss(Df[:b1, b1:], Ds[:b1, b1:]))
            losses.append(self._block_loss(Df[b1:, :b1], Ds[b1:, :b1]))
            losses.append(self._block_loss(Df[b1:, b1:], Ds[b1:, b1:]))

        loss = (
            torch.stack(losses).mean()
            if self.reduction == "mean"
            else torch.stack(losses).sum()
        )
        return loss
