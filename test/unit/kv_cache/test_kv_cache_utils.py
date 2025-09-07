import torch
from torch import nn
from joeynmt.decoders import TransformerDecoder




def rand_inputs(bs, src_len, trg_len, hidden_size):
    torch.manual_seed(11)
    encoder_output = torch.randn(bs, src_len, hidden_size)
    src_mask = torch.ones(bs, 1, src_len, dtype=torch.bool)
    trg_embed = torch.randn(bs, trg_len, hidden_size)
    trg_mask = torch.ones(bs, trg_len, 1, dtype=torch.bool)
    return encoder_output, src_mask, trg_embed, trg_mask



def build_decoder(vocab_size=17, num_layers=3, hidden_size=64, ff_size=128, num_heads=4, dropout=0.0):
    torch.manual_seed(7)
    decoder = TransformerDecoder(
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        ff_size=ff_size,
        dropout=dropout,
        emb_dropout=dropout,
        vocab_size=vocab_size,
        alpha=1.0,
        layer_norm="pre",
    )
    for p in decoder.parameters():
        if p.requires_grad:
            nn.init.uniform_(p, -0.1, 0.1)
    return decoder


def element_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


def cache_mem_bytes(cache) -> int:
    if cache is None:
        return 0
    total = 0
    for k, v in zip(cache.keys, cache.values):
        total += element_bytes(k) + element_bytes(v)
    return total