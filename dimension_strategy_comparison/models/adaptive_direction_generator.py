"""
自适应方向生成器 (Adaptive Direction Generator, ADG)
根据代码嵌入动态生成最优的投影方向
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDirectionGenerator(nn.Module):
    """
    自适应方向生成器：根据代码嵌入动态生成投影方向
    """
    def __init__(self, d=768, k=4, hidden_dims=[512, 256], use_attention=True):
        super().__init__()
        self.d = d
        self.k = k
        self.use_attention = use_attention

        # MLP Encoder
        layers = []
        in_dim = d
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Direction Generator
        self.direction_generator = nn.Linear(hidden_dims[-1], k * d)

        # Attention机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dims[-1], d),
                nn.Softmax(dim=-1)
            )

    def orthogonalize(self, W):
        """
        Gram-Schmidt正交化（可微分版本）
        输入: W (batch, k, d)
        输出: W_ortho (batch, k, d)
        """
        batch_size = W.shape[0]
        W_ortho = []

        for i in range(self.k):
            v = W[:, i, :]  # (batch, d)

            # 减去在已正交化向量上的投影
            for u in W_ortho:
                proj = (v * u).sum(dim=-1, keepdim=True)  # (batch, 1)
                v = v - proj * u

            # 归一化
            v = F.normalize(v, p=2, dim=-1)
            W_ortho.append(v)

        return torch.stack(W_ortho, dim=1)  # (batch, k, d)

    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch, d) 代码嵌入
        Returns:
            W: (batch, k, d) 正交归一化的方向矩阵
            attention_weights: (batch, d) 维度注意力权重（如果use_attention=True）
        """
        # 编码
        h = self.encoder(embeddings)  # (batch, hidden_dims[-1])

        # 生成方向
        W_flat = self.direction_generator(h)  # (batch, k*d)
        W = W_flat.view(-1, self.k, self.d)  # (batch, k, d)

        # 注意力权重
        attention_weights = None
        if self.use_attention:
            attention_weights = self.attention(h)  # (batch, d)
            # 应用注意力加权
            W = W * attention_weights.unsqueeze(1)  # (batch, k, d)

        # 正交化
        W_ortho = self.orthogonalize(W)  # (batch, k, d)

        return W_ortho, attention_weights
