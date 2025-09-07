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

from test_kv_cache_utils import build_decoder, rand_inputs
# from joeynmt.embeddings import Embeddings
# from joeynmt.vocabulary import Vocabulary
# from types import SimpleNamespace
# from joeynmt.search import greedy, beam_search, transformer_greedy, search




# 1 forward pass no cache vs. pointer
# 1 forward pass no cache vs. stack
# >1 forward passes no cache vs. pointer
# >1 forward passes no cache vs. stack

# 1 backward pass no cache vs. pointer
# 1 backward pass no cache vs. stack
# >1 backward passes no cache vs. pointer
# >1 backward passes no cache vs. stack
class TestKVCacheCorrectness(unittest.TestCase):
    def setUp(self):
        self.bs = 4
        self.src_len = 16
        self.step_len = 3
        self.hidden = 32
        self.vocab = 29
        self.layers = 2
        self.heads = 4
        self.ff = 64

    def test_forward_equivalence_no_cache_vs_pointer_single_pass(self):
        dec_a = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_b = copy.deepcopy(dec_a)

        enc_out, src_mask, trg_step, trg_mask = rand_inputs(self.bs, self.src_len, self.step_len, self.hidden)

        out_nc, _, _, _ = dec_a(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=False,
            past_key_values=None,
        )

        out_ptr, _, _, pkv = dec_b(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
        )

        self.assertIsNotNone(pkv)
        max_abs_diff = (out_nc - out_ptr).abs().max().item()
        print(f"shapes no_cache={tuple(out_nc.shape)} pointer={tuple(out_ptr.shape)} max_abs_diff={max_abs_diff:.3e}")
        torch.testing.assert_close(out_nc, out_ptr, rtol=1e-5, atol=1e-6)

    def test_forward_equivalence_no_cache_vs_stack_single_pass(self):
        dec_a = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_b = copy.deepcopy(dec_a)

        enc_out, src_mask, trg_step, trg_mask = rand_inputs(self.bs, self.src_len, self.step_len, self.hidden)

        out_nc, _, _, _ = dec_a(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=False,
            past_key_values=None,
        )

        out_stack, _, _, pkv = dec_b(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
            kv_cache_impl="stack",
        )

        self.assertIsNotNone(pkv)
        max_abs_diff = (out_nc - out_stack).abs().max().item()
        print(f"shapes no_cache={tuple(out_nc.shape)} pointer={tuple(out_stack.shape)} max_abs_diff={max_abs_diff:.3e}")
        torch.testing.assert_close(out_nc, out_stack, rtol=1e-5, atol=1e-6)

    def test_forward_equivalence_no_cache_vs_pointer_sequential_passes(self):
        torch.manual_seed(17)
        dec_ptr = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_no_cache = copy.deepcopy(dec_ptr)

        # Shared inputs
        enc_out = torch.randn(self.bs, self.src_len, self.hidden)
        src_mask = torch.ones(self.bs, 1, self.src_len, dtype=torch.bool)

        # Two sequential chunks; use fully valid masks to avoid NaNs in masked attention rows
        step1 = 4
        step2 = 5
        torch.manual_seed(18)
        trg1 = torch.randn(self.bs, step1, self.hidden)
        m1 = torch.ones(self.bs, step1, 1, dtype=torch.bool)

        torch.manual_seed(19)
        trg2 = torch.randn(self.bs, step2, self.hidden)
        m2 = torch.ones(self.bs, step2, 1, dtype=torch.bool)

        # Concatenate all inputs for no-cache processing
        all_trg = torch.cat([trg1, trg2], dim=1)  # Shape: (bs, step1+step2, hidden)
        all_mask = torch.cat([m1, m2], dim=1)     # Shape: (bs, step1+step2, 1)
        
        # Process entire sequence at once with no cache
        out_no_cache, _, _, _ = dec_no_cache(
            trg_embed=all_trg,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1 + step2,
            hidden=None,
            trg_mask=all_mask,
            use_cache=False,
            past_key_values=None,
        )

        # Process in chunks with pointer cache
        past_key_values_pointer = None
        out_pointer_chunks = []
        
        # First chunk
        out_p1, _, _, past_key_values_pointer = dec_ptr(
            trg_embed=trg1,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1,
            hidden=None,
            trg_mask=m1,
            use_cache=True,
            past_key_values=past_key_values_pointer,
            kv_cache_impl="pointer",
        )
        out_pointer_chunks.append(out_p1)
        
        # Second chunk
        out_p2, _, _, past_key_values_pointer = dec_ptr(
            trg_embed=trg2,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step2,
            hidden=None,
            trg_mask=m2,
            use_cache=True,
            past_key_values=past_key_values_pointer,
            kv_cache_impl="pointer",
        )
        out_pointer_chunks.append(out_p2)
        
        # Concatenate cached outputs
        out_pointer = torch.cat(out_pointer_chunks, dim=1)

        # Compare results
        max_abs_diff = (out_no_cache - out_pointer).abs().max().item()
        print(f"shapes no_cache={tuple(out_no_cache.shape)} pointer={tuple(out_pointer.shape)} max_abs_diff={max_abs_diff:.3e}")
        torch.testing.assert_close(out_no_cache, out_pointer, rtol=1e-5, atol=1e-6)

    def test_forward_equivalence_no_cache_vs_stack_sequential_passes(self):
        torch.manual_seed(17)
        dec_stack = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_no_cache = copy.deepcopy(dec_stack)

        # Shared inputs
        enc_out = torch.randn(self.bs, self.src_len, self.hidden)
        src_mask = torch.ones(self.bs, 1, self.src_len, dtype=torch.bool)

        # Two sequential chunks; use fully valid masks to avoid NaNs in masked attention rows
        step1 = 4
        step2 = 5
        torch.manual_seed(18)
        trg1 = torch.randn(self.bs, step1, self.hidden)
        m1 = torch.ones(self.bs, step1, 1, dtype=torch.bool)

        torch.manual_seed(19)
        trg2 = torch.randn(self.bs, step2, self.hidden)
        m2 = torch.ones(self.bs, step2, 1, dtype=torch.bool)

        # Concatenate all inputs for no-cache processing
        all_trg = torch.cat([trg1, trg2], dim=1)  # Shape: (bs, step1+step2, hidden)
        all_mask = torch.cat([m1, m2], dim=1)     # Shape: (bs, step1+step2, 1)
        
        # Process entire sequence at once with no cache
        out_no_cache, _, _, _ = dec_no_cache(
            trg_embed=all_trg,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1 + step2,
            hidden=None,
            trg_mask=all_mask,
            use_cache=False,
            past_key_values=None,
        )

        # Process in chunks with stack cache
        past_key_values_stack = None
        out_stack_chunks = []
        
        # First chunk
        out_s1, _, _, past_key_values_stack = dec_stack(
            trg_embed=trg1,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1,
            hidden=None,
            trg_mask=m1,
            use_cache=True,
            past_key_values=past_key_values_stack,
            kv_cache_impl="stack",
        )
        out_stack_chunks.append(out_s1)
        
        # Second chunk
        out_s2, _, _, past_key_values_stack = dec_stack(
            trg_embed=trg2,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step2,
            hidden=None,
            trg_mask=m2,
            use_cache=True,
            past_key_values=past_key_values_stack,
            kv_cache_impl="stack",
        )
        out_stack_chunks.append(out_s2)
        
        # Concatenate cached outputs
        out_stack = torch.cat(out_stack_chunks, dim=1)

        # Compare results
        max_abs_diff = (out_no_cache - out_stack).abs().max().item()
        print(f"shapes no_cache={tuple(out_no_cache.shape)} stack={tuple(out_stack.shape)} max_abs_diff={max_abs_diff:.3e}")
        torch.testing.assert_close(out_no_cache, out_stack, rtol=1e-5, atol=1e-6)

    def test_backward_equivalence_no_cache_vs_pointer_single_pass(self):
        dec_nc = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_ptr = copy.deepcopy(dec_nc)

        enc_out, src_mask, trg_step, trg_mask = rand_inputs(self.bs, self.src_len, self.step_len, self.hidden)

        out_nc, _, _, _ = dec_nc(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=False,
            past_key_values=None,
        )
        loss_nc = out_nc.sum()
        loss_nc.backward()
        grads_nc = [p.grad.clone() for p in dec_nc.parameters() if p.requires_grad]

        out_ptr, _, _, _ = dec_ptr(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
            kv_cache_impl="pointer",
        )
        loss_ptr = out_ptr.sum()
        loss_ptr.backward()
        grads_ptr = [p.grad.clone() for p in dec_ptr.parameters() if p.requires_grad]

        for g1, g2 in zip(grads_nc, grads_ptr):
            torch.testing.assert_close(g1, g2, rtol=1e-5, atol=1e-6)

    def test_backward_equivalence_no_cache_vs_stack_single_pass(self):
        dec_nc = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_stk = copy.deepcopy(dec_nc)

        enc_out, src_mask, trg_step, trg_mask = rand_inputs(self.bs, self.src_len, self.step_len, self.hidden)

        out_nc, _, _, _ = dec_nc(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=False,
            past_key_values=None,
        )
        loss_nc = out_nc.sum()
        loss_nc.backward()
        grads_nc = [p.grad.clone() for p in dec_nc.parameters() if p.requires_grad]

        out_stk, _, _, _ = dec_stk(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=self.step_len,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
            kv_cache_impl="stack",
        )
        loss_stk = out_stk.sum()
        loss_stk.backward()
        grads_stk = [p.grad.clone() for p in dec_stk.parameters() if p.requires_grad]

        for g1, g2 in zip(grads_nc, grads_stk):
            torch.testing.assert_close(g1, g2, rtol=1e-5, atol=1e-6)

    def test_backward_equivalence_no_cache_vs_pointer_sequential_passes(self):
        torch.manual_seed(31)
        dec_nc = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_ptr = copy.deepcopy(dec_nc)

        enc_out = torch.randn(self.bs, self.src_len, self.hidden)
        src_mask = torch.ones(self.bs, 1, self.src_len, dtype=torch.bool)

        step1 = 4
        step2 = 5
        torch.manual_seed(32)
        trg1 = torch.randn(self.bs, step1, self.hidden)
        m1 = torch.ones(self.bs, step1, 1, dtype=torch.bool)
        torch.manual_seed(33)
        trg2 = torch.randn(self.bs, step2, self.hidden)
        m2 = torch.ones(self.bs, step2, 1, dtype=torch.bool)

        # no_cache single pass over concatenated sequence to align positional encoding with cached run
        trg_cat = torch.cat([trg1, trg2], dim=1)
        m_cat = torch.cat([m1, m2], dim=1)
        out_nc_cat, _, _, _ = dec_nc(
            trg_embed=trg_cat,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1 + step2,
            hidden=None,
            trg_mask=m_cat,
            use_cache=False,
            past_key_values=None,
        )
        loss_nc = out_nc_cat.sum()
        loss_nc.backward()
        grads_nc = [p.grad.clone() for p in dec_nc.parameters() if p.requires_grad]

        # pointer sequential with cache
        out_p1, _, _, pkv = dec_ptr(
            trg_embed=trg1,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1,
            hidden=None,
            trg_mask=m1,
            use_cache=True,
            past_key_values=None,
            kv_cache_impl="pointer",
        )
        out_p2, _, _, pkv = dec_ptr(
            trg_embed=trg2,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step2,
            hidden=None,
            trg_mask=m2,
            use_cache=True,
            past_key_values=pkv,
            kv_cache_impl="pointer",
        )
        loss_ptr = (out_p1.sum() + out_p2.sum())
        loss_ptr.backward()
        grads_ptr = [p.grad.clone() for p in dec_ptr.parameters() if p.requires_grad]

        # allow tiny numerical drift in sequential cached vs no_cache
        for g1, g2 in zip(grads_nc, grads_ptr):
            torch.testing.assert_close(g1, g2, rtol=1e-4, atol=5e-6)

    def test_backward_equivalence_no_cache_vs_stack_sequential_passes(self):
        torch.manual_seed(41)
        dec_nc = build_decoder(vocab_size=self.vocab, num_layers=self.layers, hidden_size=self.hidden, ff_size=self.ff, num_heads=self.heads, dropout=0.0)
        dec_stk = copy.deepcopy(dec_nc)

        enc_out = torch.randn(self.bs, self.src_len, self.hidden)
        src_mask = torch.ones(self.bs, 1, self.src_len, dtype=torch.bool)

        step1 = 4
        step2 = 5
        torch.manual_seed(42)
        trg1 = torch.randn(self.bs, step1, self.hidden)
        m1 = torch.ones(self.bs, step1, 1, dtype=torch.bool)
        torch.manual_seed(43)
        trg2 = torch.randn(self.bs, step2, self.hidden)
        m2 = torch.ones(self.bs, step2, 1, dtype=torch.bool)

        # no_cache single pass over concatenated sequence to align positional encoding with cached run
        trg_cat = torch.cat([trg1, trg2], dim=1)
        m_cat = torch.cat([m1, m2], dim=1)
        out_nc_cat, _, _, _ = dec_nc(
            trg_embed=trg_cat,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1 + step2,
            hidden=None,
            trg_mask=m_cat,
            use_cache=False,
            past_key_values=None,
        )
        loss_nc = out_nc_cat.sum()
        loss_nc.backward()
        grads_nc = [p.grad.clone() for p in dec_nc.parameters() if p.requires_grad]

        out_s1, _, _, pkv = dec_stk(
            trg_embed=trg1,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step1,
            hidden=None,
            trg_mask=m1,
            use_cache=True,
            past_key_values=None,
            kv_cache_impl="stack",
        )
        out_s2, _, _, pkv = dec_stk(
            trg_embed=trg2,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=step2,
            hidden=None,
            trg_mask=m2,
            use_cache=True,
            past_key_values=pkv,
            kv_cache_impl="stack",
        )
        loss_stk = (out_s1.sum() + out_s2.sum())
        loss_stk.backward()
        grads_stk = [p.grad.clone() for p in dec_stk.parameters() if p.requires_grad]

        # allow tiny numerical drift in sequential cached vs no_cache
        for g1, g2 in zip(grads_nc, grads_stk):
            torch.testing.assert_close(g1, g2, rtol=1e-4, atol=5e-6)

