
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
from joeynmt.embeddings import Embeddings
from joeynmt.vocabulary import Vocabulary
from types import SimpleNamespace
from joeynmt.model import Model
from joeynmt.search import greedy, beam_search, transformer_greedy, search



# use_cache=True, no past
# use_cache=True, but past
class TestKVCacheEdgeCases(unittest.TestCase):
    def test_use_cache_true_but_no_past(self):
        dec = build_decoder(vocab_size=13, num_layers=2, hidden_size=16, ff_size=32, num_heads=4, dropout=0.0)
        enc_out, src_mask, trg_step, trg_mask = rand_inputs(bs=2, src_len=7, trg_len=3, hidden_size=16)
        _, _, _, pkv = dec(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=3,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
        )
        has_self = pkv.has_self_attention_cache()
        has_cross = pkv.has_cross_attention_cache()
        pos = pkv.get_positions_offset()
        print(f"[kv_cache.edge_case_no_past] has_cross_cache={has_cross} has_self_cache={has_self} positions_offset_shape={tuple(pos.shape)} values={pos.tolist()}")
        self.assertIsNotNone(pkv)
        self.assertTrue(has_self)
        self.assertTrue(has_cross)

    def test_have_past_but_use_cache_false(self):
        dec = build_decoder(vocab_size=13, num_layers=2, hidden_size=16, ff_size=32, num_heads=4, dropout=0.0)
        enc_out, src_mask, trg_step, trg_mask = rand_inputs(bs=2, src_len=7, trg_len=2, hidden_size=16)
        # Build a small past
        _, _, _, pkv = dec(
            trg_embed=trg_step,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=2,
            hidden=None,
            trg_mask=trg_mask,
            use_cache=True,
            past_key_values=None,
        )
        # Next call: present past, but don't extend
        trg_step2 = torch.randn(2, 1, 16)
        trg_mask2 = torch.ones(2, 1, 1, dtype=torch.bool)
        _, _, _, pkv2 = dec(
            trg_embed=trg_step2,
            encoder_output=enc_out,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=1,
            hidden=None,
            trg_mask=trg_mask2,
            use_cache=False,
            past_key_values=pkv,
        )
        # Should return the same object without modification
        same_obj = (pkv2 is pkv)
        pos1 = pkv.get_positions_offset()
        pos2 = pkv2.get_positions_offset()
        unchanged = torch.equal(pos1, pos2)
        print(f"[kv_cache.edge_case_have_past_but_no_cache] same_object={same_obj} positions_unchanged={unchanged} pos={pos2.tolist()}")
        self.assertIs(pkv2, pkv)