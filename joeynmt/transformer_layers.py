# -*- coding: utf-8 -*-
"""
Transformer layers
"""
import math
from typing import Optional

import torch
from torch import Tensor, nn

from joeynmt.builders import build_activation


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, 
                num_heads: int, 
                size: int, 
                dropout: float = 0.1,
                kv_cache_type: Optional[str] = None
                )-> None:
        """
        Create a multi-headed attention layer.

        :param num_heads: the number of heads
        :param size: hidden size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        :param kv_cache_type: type of the kv cache: None, "static" (kv cache is stored only once in the first pass), "dynamic" (updates the cache incrementally)
        """
        super().__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)


        # KV cache part (first draft)
        if kv_cache_type not in ["static", "dynamic", None]:
            raise ValueError('Cache type must be "static", "dynamic", or None.')
        self.kv_cache_type = kv_cache_type

        if self.kv_cache_type:
            self.register_buffer("k_cache", None, persistent=False)
            self.register_buffer("v_cache", None, persistent=False)

        # 1) in cross-attention kv cache is static that is computed only once
        # 2) in self-attention kv cache is dynamic that is static only for a prefix in a time step t-1 and is updated every time step


    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        k: Tensor,
        v: Tensor,
        q: Tensor,
        mask: Optional[Tensor] = None,
        return_weights: Optional[bool] = None,
        use_cache: bool = False
    ):
        """
        Computes multi-headed attention.

        :param k: keys   [batch_size, seq_len, hidden_size]
        :param v: values [batch_size, seq_len, hidden_size]
        :param q: query  [batch_size, seq_len, hidden_size]
        :param mask: optional mask [batch_size, 1, seq_len]
        :param return_weights: whether to return the attention weights,
            averaged over heads.
        :return:
            - output  [batch_size, query_len, hidden_size]
            - attention_weights  [batch_size, query_len, key_len]
        """
        batch_size = k.size(0)
        key_len = k.size(1)
        query_len = q.size(1)

        # project the queries (q)
        q = self.q_layer(q)

        # reshape q for our computation to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        ############################## KV Cache Implementation (Batch-Aware) ###################################
        # if use_cache and self.kv_cache_type:
        #     if self.kv_cache_type == "dynamic":
        #         # Dynamic cache for self-attention: build cache incrementally (batch-aware)
        #         if self.k_cache is not None:
        #             # Verify batch size consistency
        #             if self.k_cache.size(0) != batch_size:
        #                 # Batch size changed, reinitialize cache
        #                 self.k_cache = None
        #                 self.v_cache = None
                
        #         if self.k_cache is not None:
        #             # Append new tokens to existing cache
        #             new_k = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        #             new_v = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                    
        #             # If input has only 1 token (incremental), append it
        #             if new_k.size(2) == 1:
        #                 self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
        #                 self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
        #             else:
        #                 # If input has multiple tokens, replace cache entirely
        #                 self.k_cache = new_k
        #                 self.v_cache = new_v
        #         else:
        #             # Initialize cache (batch-aware)
        #             self.k_cache = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        #             self.v_cache = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                
        #         k, v = self.k_cache, self.v_cache
                
        #     elif self.kv_cache_type == "static":
        #         # Static cache for cross-attention: compute once and reuse (batch-aware)
        #         if self.k_cache is None or self.k_cache.size(0) != batch_size:
        #             # Initialize or reinitialize cache for current batch size
        #             self.k_cache = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        #             self.v_cache = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                
        #         k, v = self.k_cache, self.v_cache

        # else:
        #     k = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        #     v = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        if use_cache and self.kv_cache_type:
            if self.k_cache is not None and self.kv_cache_type == "dynamic":
                # Append new tokens to existing cache
                new_k = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                new_v = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
                self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
                
            if self.k_cache is None:
                self.k_cache = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                self.v_cache = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                
            k, v = self.k_cache, self.v_cache

        else:
            k = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
            v = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        ############################## KV Cache Implementation (Batch-Aware) ###################################

        # compute scores
        q = q / math.sqrt(self.head_size)

        # [batch_size, num_heads, query_len, key_len]
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [batch_size, 1, 1, key_len]
        if mask is not None:
            # print(f"{q.shape=}, {scores=}, {mask=}, {scores.shape=}, {mask.shape=}, {self.kv_cache_type=}")
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention_weights = self.softmax(scores)
        attention_probs = self.dropout(attention_weights)

        # get context vector (select values with attention) and reshape
        # back to [batch_size, query_len, hidden_size]
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)

        output = self.output_layer(context)

        if return_weights:
            # average attention weights over heads: [batch_size, query_len, key_len]
            attention_output_weights = attention_weights.view(
                batch_size, self.num_heads, query_len, key_len
            )
            return output, attention_output_weights.sum(dim=1) / self.num_heads
        return output, None
    
    # def forward(
    #     self,
    #     k: Tensor,
    #     v: Tensor,
    #     q: Tensor,
    #     mask: Optional[Tensor] = None,
    #     return_weights: Optional[bool] = None,
    # ):
    #     """
    #     Computes multi-headed attention.

    #     :param k: keys   [batch_size, seq_len, hidden_size]
    #     :param v: values [batch_size, seq_len, hidden_size]
    #     :param q: query  [batch_size, seq_len, hidden_size]
    #     :param mask: optional mask [batch_size, 1, seq_len]
    #     :param return_weights: whether to return the attention weights,
    #         averaged over heads.
    #     :return:
    #         - output  [batch_size, query_len, hidden_size]
    #         - attention_weights  [batch_size, query_len, key_len]
    #     """
    #     batch_size = k.size(0)
    #     key_len = k.size(1)
    #     query_len = q.size(1)

    #     # project the queries (q), keys (k), and values (v)
    #     k = self.k_layer(k)
    #     v = self.v_layer(v)
    #     q = self.q_layer(q)

    #     # reshape q, k, v for our computation to
    #     # [batch_size, num_heads, seq_len, head_dim]
    #     k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
    #     v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
    #     q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

    #     # compute scores
    #     q = q / math.sqrt(self.head_size)

    #     # [batch_size, num_heads, query_len, key_len]
    #     scores = torch.matmul(q, k.transpose(2, 3))

    #     # apply the mask (if we have one)
    #     # we add a dimension for the heads to it below: [batch_size, 1, 1, key_len]
    #     if mask is not None:
    #         scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

    #     # apply attention dropout and compute context vectors.
    #     attention_weights = self.softmax(scores)
    #     attention_probs = self.dropout(attention_weights)

    #     # get context vector (select values with attention) and reshape
    #     # back to [batch_size, query_len, hidden_size]
    #     context = torch.matmul(attention_probs, v)
    #     context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)

    #     output = self.output_layer(context)

    #     if return_weights:
    #         # average attention weights over heads: [batch_size, query_len, key_len]
    #         attention_output_weights = attention_weights.view(
    #             batch_size, self.num_heads, query_len, key_len
    #         )
    #         return output, attention_output_weights.sum(dim=1) / self.num_heads
    #     return output, None


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(
        self,
        input_size: int,
        ff_size: int,
        dropout: float = 0.1,
        alpha: float = 1.0,
        layer_norm: str = "post",
        activation: str = "relu",
    ) -> None:
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout: dropout probability
        :param alpha: weight factor for residual connection
        :param layer_norm: either "pre" or "post"
        :param activation: activation function
        """
        super().__init__()

        activation_fnc = build_activation(activation=activation)

        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            activation_fnc(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

        self.alpha = alpha
        self._layer_norm_position = layer_norm
        assert self._layer_norm_position in {"pre", "post"}

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if self._layer_norm_position == "pre":
            x = self.layer_norm(x)

        x = self.pwff_layer(x) + self.alpha * residual

        if self._layer_norm_position == "post":
            x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the input for as many time
    steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000) -> None:
        """
        Positional Encoding with maximum length

        :param size: embeddings dimension size
        :param max_len: maximum sequence length
        """
        if size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim (got dim={size})"
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, size)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb: Tensor) -> Tensor:
        """
        Embed inputs.

        :param emb: (Tensor) Sequence of word embeddings vectors
            shape (seq_len, batch_size, dim)
        :return: positionally encoded word embeddings
        """
        # Add position encodings
        return emb + self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus a position-wise
    feed-forward layer.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        alpha: float = 1.0,
        layer_norm: str = "post",
        activation: str = "relu",
    ) -> None:
        """
        A single Transformer encoder layer.

        Note: don't change the name or the order of members!
        otherwise pretrained models cannot be loaded correctly.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        :param alpha: weight factor for residual connection
        :param layer_norm: either "pre" or "post"
        :param activation: activation function
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            size,
            ff_size=ff_size,
            dropout=dropout,
            alpha=alpha,
            layer_norm=layer_norm,
            activation=activation,
        )

        self.dropout = nn.Dropout(dropout)
        self.size = size

        self.alpha = alpha
        self._layer_norm_position = layer_norm
        assert self._layer_norm_position in {"pre", "post"}

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies self attention, then dropout with residual connection (adding
        the input to the result), then layer norm, and then a position-wise
        feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        residual = x
        if self._layer_norm_position == "pre":
            x = self.layer_norm(x)

        x, _ = self.src_src_att(x, x, x, mask)
        x = self.dropout(x) + self.alpha * residual

        if self._layer_norm_position == "post":
            x = self.layer_norm(x)

        out = self.feed_forward(x)
        return out


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        alpha: float = 1.0,
        layer_norm: str = "post",
        activation: str = "relu",
        use_cache: bool = False
    ) -> None:
        """
        Represents a single Transformer decoder layer.
        It attends to the source representation and the previous decoder states.

        Note: don't change the name or the order of members!
        otherwise pretrained models cannot be loaded correctly.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        :param alpha: weight factor for residual connection
        :param layer_norm: either "pre" or "post"
        :param activation: activation function
        :param use_cache: whether to cache keys and values or not 
        """
        super().__init__()
        self.size = size

        self.use_cache = use_cache

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout, 
                                                kv_cache_type=None if not use_cache else "dynamic")
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout,
                                                kv_cache_type=None if not use_cache else "static")

        self.feed_forward = PositionwiseFeedForward(
            size,
            ff_size=ff_size,
            dropout=dropout,
            alpha=alpha,
            layer_norm=layer_norm,
            activation=activation,
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        self._layer_norm_position = layer_norm
        assert self._layer_norm_position in {"pre", "post"}

    def clear_cache(self):
        self.trg_trg_att.clear_cache()
        self.src_trg_att.clear_cache()

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
        return_attention: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        First applies target-target self-attention, dropout with residual connection
        (adding the input to the result), and layer norm.

        Second computes source-target cross-attention, dropout with residual connection
        (adding the self-attention to the result), and layer norm.

        Finally goes through a position-wise feed-forward layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as not to condition on future steps)
        :param return_attention: whether to return the attention weights
        :return:
            - output tensor
            - attention weights
        """
        # pylint: disable=unused-argument
        # 1. target-target self-attention
        residual = x
        if self._layer_norm_position == "pre":
            x = self.x_layer_norm(x)

        h1, _ = self.trg_trg_att(x, x, x, mask=trg_mask, use_cache=self.use_cache)
        h1 = self.dropout(h1) + self.alpha * residual

        if self._layer_norm_position == "post":
            h1 = self.x_layer_norm(h1)

        # 2. source-target cross-attention
        h1_residual = h1
        if self._layer_norm_position == "pre":
            h1 = self.dec_layer_norm(h1)

        h2, att = self.src_trg_att(memory, memory, h1, mask=src_mask, return_weights=return_attention, use_cache=self.use_cache)
        h2 = self.dropout(h2) + self.alpha * h1_residual

        if self._layer_norm_position == "post":
            h2 = self.dec_layer_norm(h2)

        # 3. final position-wise feed-forward layer
        out = self.feed_forward(h2)

        if return_attention:
            return out, att
        return out, None
