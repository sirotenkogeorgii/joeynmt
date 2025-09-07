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



class TestKVCacheBeam(unittest.TestCase):
    def setUp(self):
        self.bs = 8
        self.src_len = 32
        self.emb_size = 12
        self.num_layers = 2
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
        self.beam_size = 4



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


    def _build_transformer_model(self):
        # Reuse perf builder for consistent tiny model
        model = self._build_model()
        encoder_output = torch.randn(self.bs, self.src_len, self.hidden_size)
        src_mask = torch.ones(self.bs, 1, self.src_len, dtype=torch.bool)
        encoder_hidden = None
        return src_mask, model, encoder_output, encoder_hidden

    def test_beam_step1_no_cache_vs_pointer_and_stack(self):
        torch.manual_seed(0)
        src_mask, model, encoder_output, encoder_hidden = self._build_transformer_model()

        out_nc, scores_nc, _ = beam_search(
            model=model, beam_size=self.beam_size, encoder_output=encoder_output,
            encoder_hidden=encoder_hidden, src_mask=src_mask,
            max_output_length=1, alpha=0.0, n_best=self.beam_size,
            use_cache=False,
        )
        for impl in ("pointer", "stack"):
            out_cache, scores_cache, _ = beam_search(
                model=model, beam_size=self.beam_size, encoder_output=encoder_output,
                encoder_hidden=encoder_hidden, src_mask=src_mask,
                max_output_length=1, alpha=0.0, n_best=self.beam_size,
                use_cache=True, kv_cache_impl=impl,
            )
            torch.testing.assert_close(out_nc, out_cache, rtol=1e-5, atol=1e-6)
            self.assertEqual(scores_nc is None, scores_cache is None)

    def test_beam_reorder_path_runs_pointer_and_stack(self):
        torch.manual_seed(1)
        src_mask, model, encoder_output, encoder_hidden = self._build_transformer_model()

        # Two steps to trigger reorder/index_select in cache
        out_nc, scores_nc, _ = beam_search(
            model=model, beam_size=self.beam_size, encoder_output=encoder_output,
            encoder_hidden=encoder_hidden, src_mask=src_mask,
            max_output_length=5, alpha=0.0, n_best=self.beam_size,
            use_cache=False,
        )
        for impl in ("pointer", "stack"):
            out_c, scores_c, _ = beam_search(
                model=model, beam_size=self.beam_size, encoder_output=encoder_output,
                encoder_hidden=encoder_hidden, src_mask=src_mask,
                max_output_length=5, alpha=0.0, n_best=self.beam_size,
                use_cache=True, kv_cache_impl=impl,
            )
            print(f"out_nc.shape={out_nc.shape} {impl}.out_c.shape={out_c.shape}")
            self.assertEqual(out_nc.shape, out_c.shape)
            torch.testing.assert_close(out_nc, out_c, rtol=1e-5, atol=1e-6)
            self.assertEqual(scores_nc is None, scores_c is None)

    def test_beam_speed_avg(self):
        torch.manual_seed(2)
        repeats = 5
        max_output_length = 100
        src_mask, model, encoder_output, encoder_hidden = self._build_transformer_model()

        def _run(mode: str):
            total_time = 0
            # warmup
            beam_search(
                model=model, beam_size=self.beam_size, encoder_output=encoder_output,
                encoder_hidden=encoder_hidden, src_mask=src_mask,
                max_output_length=4, alpha=0.6, n_best=self.beam_size,
                use_cache=(mode != "no_cache"), kv_cache_impl=(None if mode == "no_cache" else mode),
            )
            for _ in range(repeats):
                t0 = time.perf_counter()
                beam_search(
                    model=model, beam_size=self.beam_size, encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden, src_mask=src_mask,
                    max_output_length=max_output_length, alpha=0.6, n_best=self.beam_size,
                    use_cache=(mode != "no_cache"), kv_cache_impl=(None if mode == "no_cache" else mode),
                )
                total_time += time.perf_counter() - t0
            return total_time / repeats

        mean_nc = _run("no_cache")
        mean_ptr = _run("pointer")
        mean_stk = _run("stack")

        print(f"no_cache={mean_nc:.4f}s pointer={mean_ptr:.4f}s stack={mean_stk:.4f}s")
        # Allow some slack; caches should not be significantly slower
        self.assertLessEqual(mean_ptr, mean_nc)
        self.assertLessEqual(mean_stk, mean_nc)