import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, args, causal=False):
        super(Attention, self).__init__()
        self.args = args
        self.base_dim = args.base_dim
        self.prefer_dim = args.prefer_dim
        self.causal = causal

        self.w_k = nn.Linear(self.base_dim, self.base_dim+self.prefer_dim)
        self.w_v = nn.Linear(self.base_dim, self.base_dim)

    def forward(self, q, k):
        v = k
        _, seq_len, _ = q.shape

        if self.causal:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=q.device, dtype=q.dtype),
                diagonal=1
            )
        query_i = q
        key_i = self.w_k(k)
        value_i = v
        attn_weights = torch.matmul(query_i, key_i.transpose(-2, -1))
        scale = 1.0 / (key_i.size(-1) ** 0.5)
        attn_weights_scaled = attn_weights * scale
        if self.causal:
            attn_weights_scaled = attn_weights_scaled + causal_mask
        attn_scores = torch.softmax(attn_weights_scaled, dim=-1)
        attn_output = torch.matmul(attn_scores, value_i)

        return attn_output, attn_weights
