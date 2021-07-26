"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, ret_attn_scores):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.ret_attn_scores = ret_attn_scores
        
    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        # this is what's used to visualize attention
        scores = self.drop(F.softmax(scores, dim=-1)) 
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        if self.ret_attn_scores:
            return h, scores
        else:
            return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, hidden_dropout_prob, 
    attention_probs_dropout_prob, layer_norm_eps, ret_attn_scores):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, attention_probs_dropout_prob, ret_attn_scores)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.drop = nn.Dropout(hidden_dropout_prob)
        self.ret_attn_scores = ret_attn_scores
        
    def forward(self, x, mask):
        if self.ret_attn_scores:
            h, scores = self.attn(self.norm1(x), mask) # eq 1
        else:
            h = self.attn(self.norm1(x), mask) # eq 1
        h = self.drop(self.proj(h)) # eq 1
        x = x + h # eq 2
        h = self.drop(self.pwff(self.norm2(x))) # eq 3
        x = x + h # eq 3
        if self.ret_attn_scores:
            return x, scores
        else:
            return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, hidden_dropout_prob, 
    attention_probs_dropout_prob, layer_norm_eps, ret_attn_scores, ret_interm_repr):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, hidden_dropout_prob, 
            attention_probs_dropout_prob, layer_norm_eps, ret_attn_scores) for _ in range(num_layers)])
        
        self.ret_attn_scores = ret_attn_scores
        self.ret_interm_repr = ret_interm_repr

    def forward(self, x, mask=None):
        if self.ret_attn_scores:
            scores_list = []
        if self.ret_interm_repr:
            interm_repr_list = []

        for block in self.blocks:
            if self.ret_attn_scores:
                x, scores = block(x, mask)
                scores_list.append(scores)
            else:
                x = block(x, mask)
            if self.ret_interm_repr:
                interm_repr_list.append(x)

        if self.ret_interm_repr and self.ret_attn_scores:
            return x, interm_repr_list, scores_list
        elif self.ret_interm_repr:
            return x, interm_repr_list
        elif self.ret_attn_scores:
            return x, scores_list
        else:
            return x
