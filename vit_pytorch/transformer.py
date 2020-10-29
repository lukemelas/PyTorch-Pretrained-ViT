"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import copy 
from typing import Optional, Any
from torch import nn
from torch import Tensor 
from torch.nn import functional as F


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.n_heads = cfg.n_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.drop(self.attn(self.norm1(x), mask))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        for block in self.blocks:
            h = block(h, mask)
        return h


# 
    # def _get_clones(module, N):
    #     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


    # def _get_activation_fn(activation):
    #     if activation == "relu":
    #         return F.relu
    #     elif activation == "gelu":
    #         return F.gelu
    #     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


    # class TransformerEncoderLayer(nn.Module):
    #     r"""A modified TransformerEncoderLayer based on the PyTorch implementation. 

    #     Args:
    #         d_model: the number of expected features in the input (required).
    #         nhead: the number of heads in the multiheadattention models (required).
    #         dim_feedforward: the dimension of the feedforward network model (default=2048).
    #         dropout: the dropout value (default=0.1).
    #         activation: the activation function of intermediate layer, relu or gelu (default=relu).

    #     Examples::
    #         >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
    #         >>> src = torch.rand(10, 32, 512)
    #         >>> out = encoder_layer(src)
    #     """

    #     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
    #         super(TransformerEncoderLayer, self).__init__()
    #         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    #         self.linear1 = nn.Linear(d_model, dim_feedforward)
    #         self.dropout = nn.Dropout(dropout)
    #         self.linear2 = nn.Linear(dim_feedforward, d_model)

    #         self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
    #         self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
    #         self.dropout1 = nn.Dropout(dropout)
    #         self.dropout2 = nn.Dropout(dropout)

    #         self.activation = _get_activation_fn(activation)

    #     def __setstate__(self, state):
    #         if 'activation' not in state:
    #             state['activation'] = F.relu
    #         super(TransformerEncoderLayer, self).__setstate__(state)

    #     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
    #                 src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    #         r"""Pass the input through the encoder layer.

    #         Args:
    #             src: the sequence to the encoder layer (required).
    #             src_mask: the mask for the src sequence (optional).
    #             src_key_padding_mask: the mask for the src keys per batch (optional).

    #         Shape:
    #             see the docs in Transformer class.
    #         """
    #         src2 = self.norm1(src)
    #         src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
    #         src2 = self.dropout1(src2)
    #         src = src2 + src
    #         src2 = self.norm2(src2)
    #         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
    #         src2 = self.dropout2(src2)
    #         src = src2 + src
    #         return src


    # class TransformerEncoder(nn.Module):
    #     r"""TransformerEncoder is a stack of N encoder layers

    #     Args:
    #         encoder_layer: an instance of the TransformerEncoderLayer() class (required).
    #         num_layers: the number of sub-encoder-layers in the encoder (required).
    #         norm: the layer normalization component (optional).

    #     Examples::
    #         >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
    #         >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
    #         >>> src = torch.rand(10, 32, 512)
    #         >>> out = transformer_encoder(src)
    #     """
    #     __constants__ = ['norm']

    #     def __init__(self, encoder_layer, num_layers, norm=None):
    #         super(TransformerEncoder, self).__init__()
    #         self.layers = _get_clones(encoder_layer, num_layers)
    #         self.num_layers = num_layers
    #         self.norm = norm

    #     def forward(self, src: Tensor, mask: Optional[Tensor] = None, 
    #                 src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    #         r"""Pass the input through the encoder layers in turn.

    #         Args:
    #             src: the sequence to the encoder (required).
    #             mask: the mask for the src sequence (optional).
    #             src_key_padding_mask: the mask for the src keys per batch (optional).

    #         Shape:
    #             see the docs in Transformer class.
    #         """
    #         output = src

    #         for mod in self.layers:
    #             output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

    #         if self.norm is not None:
    #             output = self.norm(output)

    #         return output