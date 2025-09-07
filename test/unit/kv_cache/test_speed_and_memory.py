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



# speed > forward passes no cache vs. pointer
# speed > forward passes no cache vs. stack
# memory 2 forward passes pointer vs. stack
class TestKVCachePerfAndMemory(unittest.TestCase):
    def setUp(self):
        self.bs = 8
        self.src_len = 32
        self.emb_size = 12
        self.num_layers = 2
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0



        special_symbols = SimpleNamespace(**{
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "sep_token": "<sep>",
            "unk_id": 0,
            "pad_id": 1,
            "bos_id": 2,
            "eos_id": 3,
            "sep_id": 4,
            "lang_tags": ["<de>", "<en>"],
        })
        self.vocab = Vocabulary(tokens=["word"], cfg=special_symbols)
        self.vocab_size = len(self.vocab)

    def _build_model(self):
        """Helper to build a model with or without KV cache."""
        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
            padding_idx=1,
        )
        
        decoder = TransformerDecoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            vocab_size=self.vocab_size,
            layer_norm="pre"
        )
        
        model = Model(
            encoder=None,
            decoder=decoder,
            src_embed=emb,
            trg_embed=emb,
            src_vocab=self.vocab,
            trg_vocab=self.vocab,
        )
        
        # Initialize weights consistently
        for p in model.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)
            
        return model

    def speed_up_measurement(self, cache_implementation, num_runs=5):
        max_output_length = 120
        
        # Build models with identical weights
        model_no_cache = self._build_model()
        model_with_cache = self._build_model()
        
        for param_no_cache, param_with_cache in zip(
            model_no_cache.parameters(), model_with_cache.parameters()
        ):
            param_with_cache.data.copy_(param_no_cache.data)
        
        encoder_output = torch.rand(size=(self.bs, self.src_len, self.hidden_size))
        src_mask = torch.ones(size=(self.bs, 1, self.src_len)) == 1
        
        # Warmup runs to ensure fair timing
        for _ in range(2):
            _ = transformer_greedy(
                src_mask=src_mask, max_output_length=10, model=model_no_cache,
                encoder_output=encoder_output, encoder_hidden=None, use_cache=False
            )
            _ = transformer_greedy(
                src_mask=src_mask, max_output_length=10, model=model_with_cache,
                encoder_output=encoder_output, encoder_hidden=None, 
                use_cache=True,
                kv_cache_impl=cache_implementation
            )
        
        # Measure time without cache (average per iteration)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        for _ in range(num_runs):
            output_no_cache, _, _ = transformer_greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                min_output_length=max_output_length,
                model=model_no_cache,
                encoder_output=encoder_output,
                encoder_hidden=None,
                use_cache=False,
                return_attention=False,
                return_prob="none"
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time_no_cache = time.perf_counter() - start_time
        time_no_cache = total_time_no_cache / num_runs  # Average time per iteration
        
        # Measure time with cache (average per iteration)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        for _ in range(num_runs):
            output_with_cache, _, _ = transformer_greedy(
                src_mask=src_mask,
                max_output_length=max_output_length,
                min_output_length=max_output_length,
                model=model_with_cache,
                encoder_output=encoder_output,
                encoder_hidden=None,
                use_cache=True,
                kv_cache_impl=cache_implementation,
                return_attention=False,
                return_prob="none"
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time_with_cache = time.perf_counter() - start_time
        time_with_cache = total_time_with_cache / num_runs  # Average time per iteration

        return time_no_cache, output_no_cache, time_with_cache, output_with_cache

    def test_speedup_with_pointer_cache(self):
        time_no_cache, output_no_cache, time_with_cache, output_with_cache = self.speed_up_measurement("pointer", num_runs=10)
        torch.testing.assert_close(output_no_cache, output_with_cache, rtol=1e-4, atol=1e-4)
        speedup = time_no_cache / max(time_with_cache, 1e-9)
        print(f"No cache: {time_no_cache:.4f}s. With cache: {time_with_cache:.4f}s. Speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 1.0, "KV cache should not significantly degrade performance")

    def test_speedup_with_stack_cache(self):
        time_no_cache, output_no_cache, time_with_cache, output_with_cache = self.speed_up_measurement("stack", num_runs=10)
        torch.testing.assert_close(output_no_cache, output_with_cache, rtol=1e-4, atol=1e-4)
        speedup = time_no_cache / max(time_with_cache, 1e-9)
        print(f"No cache: {time_no_cache:.4f}s. With cache: {time_with_cache:.4f}s. Speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 1.0, "KV cache should not significantly degrade performance")

    def test_memory_comparison_pointer_vs_stack(self):
        # Prepare a first step with pointer, then build stack from it, then do another step and compare memory
        decoder = build_decoder(vocab_size=self.vocab_size, num_layers=self.num_layers, hidden_size=self.hidden_size, ff_size=self.ff_size, num_heads=self.num_heads, dropout=0.0)
        enc_out = torch.randn(self.bs, self.src_len, self.hidden_size)
        src_mask = torch.ones(self.bs, 1, self.src_len, dtype=torch.bool)

        # First step, create pointer cache
        step_len = 3
        trg_embed = torch.randn(self.bs, step_len, self.hidden_size)
        trg_mask = torch.ones(self.bs, step_len, 1, dtype=torch.bool)
        _, _, _, pkv_ptr = decoder(
            trg_embed=trg_embed,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
        )

        # Build stack cache with identical data
        self_layers = [pkv_ptr.get_self(i) for i in range(len(decoder.layers))]
        # ptr_self = KVCachePointer(self_layers, valid_token_mask=torch.ones(self.bs, step_len, dtype=torch.bool))
        stk_self = KVCacheStack(self_layers, valid_token_mask=torch.ones(self.bs, step_len, dtype=torch.bool))
        pkv_stack = EncoderDecoderCache(self_attention_cache=stk_self, cross_attention_cache=pkv_ptr.cross_attention_cache)

        # Next heterogeneous step: vary valid tokens per batch
        step_len2 = 5
        trg_embed2 = torch.randn(self.bs, step_len2, self.hidden_size)
        # Make half of the batch use fewer valid positions to simulate ragged growth
        valid_counts = torch.randint(low=1, high=step_len2 + 1, size=(self.bs,))
        trg_mask2 = torch.zeros(self.bs, step_len2, 1, dtype=torch.bool)
        for b in range(self.bs):
            trg_mask2[b, :valid_counts[b], 0] = True

        # Update pointer
        _, _, _, pkv_ptr2 = decoder(
            trg_embed=trg_embed2,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step_len2,
            hidden=None,
            trg_mask=trg_mask2,
            use_cache=True,
            past_key_values=pkv_ptr,
        )
        # Update stack
        _, _, _, pkv_stack2 = decoder(
            trg_embed=trg_embed2,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step_len2,
            hidden=None,
            trg_mask=trg_mask2,
            use_cache=True,
            past_key_values=pkv_stack,
        )

        mem_ptr = cache_mem_bytes(pkv_ptr2.self_attention_cache)
        mem_stk = cache_mem_bytes(pkv_stack2.self_attention_cache)

        print(f"pointer_bytes={mem_ptr} stack_bytes={mem_stk} delta={mem_ptr - mem_stk}")

        # Pointer should be less or equal memory than stack in heterogeneous updates
        self.assertLessEqual(mem_ptr, mem_stk)