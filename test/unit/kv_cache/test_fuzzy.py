import unittest
import time
import copy
import random

import torch
from torch import nn

from joeynmt.decoders import TransformerDecoder
from joeynmt.kv_cache import (
    KVCacheStack,
    KVCachePointer,
    EncoderDecoderCache,
)

from test_kv_cache_utils import build_decoder, rand_inputs, cache_mem_bytes
from joeynmt.embeddings import Embeddings
from joeynmt.vocabulary import Vocabulary
from types import SimpleNamespace
from joeynmt.model import Model
from joeynmt.search import greedy, beam_search, transformer_greedy, search



# memory >2 forward passes pointer vs. stack (long sequence (input) length)
class TestKVCacheFuzzy(unittest.TestCase):
    def test_memory_comparison_pointer_vs_stack_fuzzy(self):
        torch.manual_seed(5)
        # Longer input length for cross-attention cache realism
        src_len = 24
        batch_size = 6
        hidden_size = 32
        num_layers = 3
        num_heads = 4
        ff_size = 64

        # Decoder and inputs
        decoder = build_decoder(
            vocab_size=29,
            num_layers=num_layers,
            hidden_size=hidden_size,
            ff_size=ff_size,
            num_heads=num_heads,
            dropout=0.0,
        )
        enc_out = torch.randn(batch_size, src_len, hidden_size)
        src_mask = torch.ones(batch_size, 1, src_len, dtype=torch.bool)

        # First step: create pointer cache with valid step_len > 10
        step_len_initial = 12
        trg_embed = torch.randn(batch_size, step_len_initial, hidden_size)
        trg_mask = torch.ones(batch_size, step_len_initial, 1, dtype=torch.bool)
        _, _, _, pkv_ptr = decoder(
            trg_embed=trg_embed,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step_len_initial,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
        )

        # Build stack cache with identical data
        self_layers = [pkv_ptr.get_self(i) for i in range(len(decoder.layers))]
        valid_mask_init = torch.ones(batch_size, step_len_initial, dtype=torch.bool)
        stk_self = KVCacheStack(self_layers, valid_token_mask=valid_mask_init)
        pkv_stack = EncoderDecoderCache(
            self_attention_cache=stk_self,
            cross_attention_cache=pkv_ptr.cross_attention_cache,
        )

        # Multiple sequential decoding iterations with ragged growth per batch
        num_iters = 100
        max_step = 11  # each iteration adds up to 11 tokens (>10 requested overall across passes)
        for _ in range(num_iters):
            step_len = max_step
            trg_embed_i = torch.randn(batch_size, step_len, hidden_size)

            # Randomly make about half of batch use fewer valid positions
            # Build a per-batch valid count in [1, step_len]
            valid_counts = torch.randint(low=1, high=step_len + 1, size=(batch_size,))
            trg_mask_i = torch.zeros(batch_size, step_len, 1, dtype=torch.bool)
            for b in range(batch_size):
                keep = valid_counts[b].item()
                trg_mask_i[b, :keep, 0] = True

            # Update pointer cache
            _, _, _, pkv_ptr = decoder(
                trg_embed=trg_embed_i,
                encoder_output=enc_out,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=step_len,
                hidden=None,
                trg_mask=trg_mask_i,
                use_cache=True,
                past_key_values=pkv_ptr,
            )

            # Update stack cache
            _, _, _, pkv_stack = decoder(
                trg_embed=trg_embed_i,
                encoder_output=enc_out,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=step_len,
                hidden=None,
                trg_mask=trg_mask_i,
                use_cache=True,
                past_key_values=pkv_stack,
            )

        mem_ptr = cache_mem_bytes(pkv_ptr.self_attention_cache)
        mem_stk = cache_mem_bytes(pkv_stack.self_attention_cache)
        print(
            f"[kv_cache.fuzzy_memory] iters={num_iters} step_len={step_len} src_len={src_len} "
            f"pointer_bytes={mem_ptr} stack_bytes={mem_stk} delta={mem_ptr - mem_stk} factor={mem_ptr / mem_stk:.2f}x"
        )

        # Pointer should be less or equal memory than stack under ragged growth
        self.assertLessEqual(mem_ptr, mem_stk)

