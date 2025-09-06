from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import torch
from torch import Tensor, nn
from enum import Enum



class KVCacheImpl(str, Enum):
    pointer = "pointer"
    stack = "stack"



class KVCache(ABC):
    keys: List[Tensor]  # len=num_layers, [bs, num_heads, cache_len, dim]
    values: List[Tensor]  # len=num_layers, [bs, num_heads, cache_len, dim]

    batch_size: int
    num_layers: int
    cache_len: int
    # Number of valid cached time steps per batch element (used for masking and PE shift)
    cache_positions: Tensor  # [bs], dtype=torch.long

    def __init__(
        self,
        layers_kv: List[Tuple[Tensor, Tensor]], # [num_layers, 2, batch_size, num_heads, seq_len, head_dim]
        valid_token_mask: Optional[Tensor] = None,  # [bs, seq_len] bool
    ):
        """
        Initialize KV cache from a list of per-layer (k_new, v_new) tensors.

        Each (k_new, v_new) has shape [batch_size, num_heads, seq_len, head_dim].
        For dynamic caches, we'll allow future extension along seq_len.
        For static caches, entries are fixed and won't be updated.
        """
        assert isinstance(layers_kv, list) and len(layers_kv) > 0
        k0, v0 = layers_kv[0]
        assert k0.dim() == 4 and v0.dim() == 4, (k0.shape, v0.shape)
        assert k0.shape == v0.shape

        self.batch_size = k0.size(0)
        self.num_layers = len(layers_kv)
        num_heads = k0.size(1)
        init_len = k0.size(2)
        head_dim = k0.size(3)
        device = k0.device
        dtype = k0.dtype

        self.cache_len = init_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        # # cache_positions counts how many valid cached steps exist per batch
        # # For newly created cache, we consider all provided steps valid
        # if valid_token_mask is not None:
        #     assert valid_token_mask.dim() == 2 # mask is supposed to be 2D

        #     if valid_token_mask.shape[0] != self.batch_size: # wrong batch size of the valid tokens mask
        #         raise AssertionError(f"valid_token_mask batch dim {valid_token_mask.shape[0]} != {self.batch_size}")
            
        #     assert valid_token_mask.shape[1] == init_len # mask has a valid length
        #     self.cache_positions = valid_token_mask.to(dtype=torch.long, device=device).sum(dim=1)
        # else:
        #     # valid tokens mask is not provided, meaning that all tokens are valid (assumption)
        #     self.cache_positions = torch.full((self.batch_size,), init_len, dtype=torch.long, device=device)

        # Store rectangular tensors per layer; copy the provided initial content
        self.keys = []
        self.values = []
        for (k_new, v_new) in layers_kv:
            assert k_new.shape == (self.batch_size, num_heads, init_len, head_dim)
            assert v_new.shape == (self.batch_size, num_heads, init_len, head_dim)
            self.keys.append(k_new.clone())
            self.values.append(v_new.clone())

    @abstractmethod
    def get_positions_offset(self):
        pass

    @abstractmethod
    def get_active_cache_mask(self):
        pass

    @abstractmethod
    def index_select(self, index: Tensor):
        pass

    def get_layer(self, layer_i):
        if not (0 <= layer_i < self.num_layers):
            raise IndexError(f"Layer index out of range: {layer_i}")
        return self.keys[layer_i], self.values[layer_i]
    


class MutableKVCache(KVCache):
    @abstractmethod
    def update_cache(self, new_self_kv, valid_token_mask_new: Optional[Tensor] = None,  # [bs, seq_new] bool indicating valid timesteps per batch
    ) -> None:
        pass



class StaticKVCache(KVCache):
    valid_token_mask: Tensor
    positions_offset: Tensor

    def __init__(
        self,
        layers_kv: List[Tuple[Tensor, Tensor]], # [num_layers, 2, batch_size, num_heads, seq_len, head_dim]
        valid_token_mask: Optional[Tensor] = None,  # [bs, seq_len] bool
    ):
        """
        Initialize KV cache from a list of per-layer (k_new, v_new) tensors.

        Each (k_new, v_new) has shape [batch_size, num_heads, seq_len, head_dim].
        For dynamic caches, we'll allow future extension along seq_len.
        For static caches, entries are fixed and won't be updated.
        """
        super().__init__(layers_kv, valid_token_mask)
        
        # self.batch_size = k0.size(0)
        # self.num_layers = len(layers_kv)
        # num_heads = k0.size(1)
        # init_len = k0.size(2)
        # head_dim = k0.size(3)
        device = layers_kv[0][0].device
        # dtype = k0.dtype

        if valid_token_mask is not None:
            assert valid_token_mask.dim() == 2

            if valid_token_mask.size(0) != self.batch_size:
                raise AssertionError(f"valid_token_mask batch dim {valid_token_mask.shape[0]} != {self.batch_size}")
            assert valid_token_mask.size(1) == self.cache_len
            self.positions_offset = valid_token_mask.to(dtype=torch.long, device=device).sum(dim=1)
        else:
            # If mask is not provided, assume all initial tokens are valid
            self.positions_offset = torch.full((self.batch_size,), self.cache_len, dtype=torch.long, device=device)


    def get_positions_offset(self):
        return self.positions_offset
    
    def get_active_cache_mask(self) -> Tensor:
        """Return [bs, cache_len] bool mask: True for active cached positions."""
        device = self.keys[0].device
        arange = torch.arange(self.cache_len, device=device)
        return arange[None, :] < self.positions_offset[:, None]

    @property
    def valid_token_mask(self) -> Tensor:
        """Compatibility: compute mask on-the-fly from positions_offset."""
        return self.get_active_cache_mask()

    def index_select(self, index: Tensor) -> "KVCacheStack":
        """
        Return a new KVCacheStack reindexed along the batch dimension using `index`.

        index: 1D LongTensor of shape [new_bs]
        """
        assert index.dim() == 1
        # Reindex per-layer keys/values
        new_layers_kv: List[Tuple[Tensor, Tensor]] = []
        for k, v in zip(self.keys, self.values):
            k_sel = k.index_select(0, index)
            v_sel = v.index_select(0, index)
            new_layers_kv.append((k_sel, v_sel))

        # Reindex validity mask (and thereby positions_offset)
        valid_mask_sel = self.valid_token_mask.index_select(0, index) if hasattr(self, 'valid_token_mask') else None

        return StaticKVCache(new_layers_kv, valid_token_mask=valid_mask_sel)    




class KVCacheStack(MutableKVCache):
    valid_token_mask: Tensor
    positions_offset: Tensor

    def __init__(
        self,
        layers_kv: List[Tuple[Tensor, Tensor]], # [num_layers, 2, batch_size, num_heads, seq_len, head_dim]
        valid_token_mask: Optional[Tensor] = None,  # [bs, seq_len] bool
    ):
        """
        Initialize KV cache from a list of per-layer (k_new, v_new) tensors.

        Each (k_new, v_new) has shape [batch_size, num_heads, seq_len, head_dim].
        For dynamic caches, we'll allow future extension along seq_len.
        For static caches, entries are fixed and won't be updated.
        """
        super().__init__(layers_kv, valid_token_mask)
        
        # self.batch_size = k0.size(0)
        # self.num_layers = len(layers_kv)
        # num_heads = k0.size(1)
        # init_len = k0.size(2)
        # head_dim = k0.size(3)
        device = layers_kv[0][0].device
        # dtype = k0.dtype

        if valid_token_mask is not None:
            assert valid_token_mask.dim() == 2 # mask is supposed to be 2D

            if valid_token_mask.size(0) != self.batch_size: # wrong batch size of the valid tokens mask
                raise AssertionError(f"valid_token_mask batch dim {valid_token_mask.shape[0]} != {self.batch_size}")
            
            assert valid_token_mask.size(1) == self.cache_len # mask has a valid length
            self.valid_token_mask = valid_token_mask.to(device=device)
        else:
            # valid tokens mask is not provided, meaning that all tokens are valid (assumption)
            self.valid_token_mask = torch.ones((self.batch_size, self.cache_len), device=device).bool()
        self.positions_offset = self.valid_token_mask.to(dtype=torch.long, device=device).sum(dim=1)
        # compatibility alias for older callers
        self.cache_positions = self.positions_offset

    def get_positions_offset(self):
        return self.positions_offset

    def get_active_cache_mask(self) -> Tensor:
        """Return [bs, cache_len] bool mask: True for active cached positions."""
        # device = self.keys[0].device
        # arange = torch.arange(self.cache_len, device=device)
        # return arange[None, :] < self.cache_positions[:, None]
        return self.valid_token_mask

    def update_cache(self, new_self_kv, valid_token_mask_new):
        # print("KVCacheStack")
        bs, num_heads, seq_new, head_dim = new_self_kv[0][0].shape

        if valid_token_mask_new is not None:
            assert valid_token_mask_new.shape[:2] == (bs, seq_new)
            valid_token_mask_new = valid_token_mask_new.to(dtype=torch.bool)
        else:
            valid_token_mask_new = torch.ones((self.batch_size, seq_new), device=new_self_kv[0][0].device).bool()

        new_offsets = valid_token_mask_new.long().sum(dim=1)  # [bs]
        self.positions_offset += new_offsets
        self.valid_token_mask = torch.cat([self.valid_token_mask, valid_token_mask_new], dim=1)
        self.cache_len += seq_new
        # keep legacy field in sync
        self.cache_positions = self.positions_offset

        for layer_i, (k_new, v_new) in enumerate(new_self_kv):
            self._update_layer(layer_i, k_new, v_new, valid_token_mask_new=None)


    def _update_layer(self, layer_i: int, k_new: Tensor, v_new: Tensor,
        valid_token_mask_new: Optional[Tensor] = None,  # [bs, seq_new] bool indicating valid timesteps per batch
    ) -> None:
        """
        Append new KV entries for a given layer.

        k_new, v_new: [bs, num_heads, seq_new, head_dim]
        For static caches, this is a no-op (kept for API completeness).
        """
        
        assert 0 <= layer_i < self.num_layers
        assert k_new.shape == v_new.shape
        assert k_new.dim() == 4
        bs, num_heads, seq_new, head_dim = k_new.shape
        assert bs == self.batch_size
        assert head_dim == self.head_dim
        assert num_heads == self.num_heads

        self.keys[layer_i] = torch.cat([self.keys[layer_i], k_new], dim=2)
        self.values[layer_i] = torch.cat([self.values[layer_i], v_new], dim=2)
    

    def index_select(self, index: Tensor) -> "KVCacheStack":
        """
        Return a new KVCacheStack reindexed along the batch dimension using `index`.

        index: 1D LongTensor of shape [new_bs]
        """
        assert index.dim() == 1
        # Reindex per-layer keys/values
        new_layers_kv: List[Tuple[Tensor, Tensor]] = []
        for k, v in zip(self.keys, self.values):
            k_sel = k.index_select(0, index)
            v_sel = v.index_select(0, index)
            new_layers_kv.append((k_sel, v_sel))

        # Reindex validity mask (and thereby positions_offset)
        valid_mask_sel = self.valid_token_mask.index_select(0, index) if hasattr(self, 'valid_token_mask') else None

        return KVCacheStack(new_layers_kv, valid_token_mask=valid_mask_sel)
    



class KVCachePointer(MutableKVCache):
    positions_offset: Tensor

    def __init__(
        self,
        layers_kv: List[Tuple[Tensor, Tensor]], # [num_layers, 2, batch_size, num_heads, seq_len, head_dim]
        valid_token_mask: Optional[Tensor] = None,  # [bs, seq_len] bool
    ):
        """
        Initialize KV cache from a list of per-layer (k_new, v_new) tensors.

        Each (k_new, v_new) has shape [batch_size, num_heads, seq_len, head_dim].
        For dynamic caches, we'll allow future extension along seq_len.
        For static caches, entries are fixed and won't be updated.
        """
        super().__init__(layers_kv, valid_token_mask)
        
        # self.batch_size = k0.size(0)
        # self.num_layers = len(layers_kv)
        # num_heads = k0.size(1)
        # init_len = k0.size(2)
        # head_dim = k0.size(3)
        device = layers_kv[0][0].device
        # dtype = k0.dtype

        if valid_token_mask is not None:
            assert valid_token_mask.dim() == 2 # mask is supposed to be 2D

            if valid_token_mask.size(0) != self.batch_size: # wrong batch size of the valid tokens mask
                raise AssertionError(f"valid_token_mask batch dim {valid_token_mask.shape[0]} != {self.batch_size}")
            
            assert valid_token_mask.size(1) == self.cache_len # mask has a valid length
            self.positions_offset = valid_token_mask.to(dtype=torch.long, device=device).sum(dim=1)
        else:
            # valid tokens mask is not provided, meaning that all tokens are valid (assumption)
            self.positions_offset = torch.full((self.batch_size,), self.cache_len, dtype=torch.long, device=device)
        # compatibility alias for older callers
        self.cache_positions = self.positions_offset


    def index_select(self, index: Tensor) -> "KVCachePointer":
        """
        Return a new KVCachePointer reindexed along the batch dimension using `index`.

        index: 1D LongTensor of shape [new_bs]
        """
        assert index.dim() == 1
        # Reindex per-layer keys/values
        new_layers_kv: List[Tuple[Tensor, Tensor]] = []
        for k, v in zip(self.keys, self.values):
            k_sel = k.index_select(0, index)
            v_sel = v.index_select(0, index)
            new_layers_kv.append((k_sel, v_sel))

        # Create new cache and preserve positions_offset
        new_cache = KVCachePointer(new_layers_kv, valid_token_mask=None)
        new_cache.positions_offset = self.positions_offset.index_select(0, index)
        return new_cache

    def get_positions_offset(self):
        return self.positions_offset

    
    def update_cache(self, new_self_kv, valid_token_mask_new):
        # print("KVCachePointer")
        bs, num_heads, seq_new, head_dim = new_self_kv[0][0].shape

        if valid_token_mask_new is not None:
            assert valid_token_mask_new.shape[:2] == (bs, seq_new)
            valid_token_mask_new = valid_token_mask_new.to(dtype=torch.bool)
            new_lengths = valid_token_mask_new.long().sum(dim=1)  # [bs]
        else:
            valid_token_mask_new = torch.ones((self.batch_size, seq_new), device=new_self_kv[0][0].device).bool()
            new_lengths = torch.full((bs,), seq_new, dtype=torch.long, device=new_self_kv[0][0].device)

        # Ensure capacity to accommodate writes
        needed_len = int((self.positions_offset + new_lengths).max().item())
        self._ensure_capacity(needed_len)

        # Write per layer (vectorized)
        for layer_i, (k_new, v_new) in enumerate(new_self_kv):
            self._update_layer(layer_i, k_new, v_new, valid_token_mask_new=valid_token_mask_new)

        # Advance positions
        self.positions_offset = self.positions_offset + new_lengths.to(self.positions_offset.device)
        # compatibility alias update
        self.cache_positions = self.positions_offset


    def _update_layer(
        self,
        layer_i: int,
        k_new: Tensor,
        v_new: Tensor,
        valid_token_mask_new: Optional[Tensor] = None,  # [bs, seq_new] bool
    ) -> None:
        """
        Write new KV entries for a given layer at positions starting from positions_offset.
        k_new, v_new: [bs, num_heads, seq_new, head_dim]
        """
        assert 0 <= layer_i < self.num_layers
        assert k_new.shape == v_new.shape and k_new.dim() == 4
        bs, num_heads, seq_new, head_dim = k_new.shape
        assert bs == self.batch_size and num_heads == self.num_heads and head_dim == self.head_dim

        device = k_new.device
        assert self.cache_len > 0, "cache_len must be > 0"

        # Time-valid mask [bs, seq_new]
        if valid_token_mask_new is None:
            time_valid = torch.ones((bs, seq_new), dtype=torch.bool, device=device)
        else:
            assert valid_token_mask_new.shape == (bs, seq_new)
            time_valid = valid_token_mask_new.to(dtype=torch.bool, device=device)

        
        # NOTE: one-liner that is trying to replace the code above. Remove if unsuccessful
        # target_idx = self.positions_offset.to(device)[:, None] + torch.arange(seq_new)[None, :]

        # Flatten batch*heads for scatter
        bh = bs * num_heads
        k_dest = self.keys[layer_i]
        v_dest = self.values[layer_i]
        k_dest_flat = k_dest.reshape(bh, self.cache_len, head_dim)
        v_dest_flat = v_dest.reshape(bh, self.cache_len, head_dim)

        k_src_flat = k_new.reshape(bh, seq_new, head_dim)
        v_src_flat = v_new.reshape(bh, seq_new, head_dim)

        all_valid = (valid_token_mask_new is None) or valid_token_mask_new.all()
        starts = self.positions_offset  # [bs]

        if all_valid:
            # Case 1: all batches aligned → one big contiguous copy
            if torch.equal(starts, starts[0].expand_as(starts)):
                s = int(starts[0].item())
                k_dest_flat[:, s:s+seq_new, :].copy_(k_src_flat)
                v_dest_flat[:, s:s+seq_new, :].copy_(v_src_flat)
            else:
                # Case 2: few distinct starts → group and copy per group
                uniq, inv = torch.unique(starts, return_inverse=True)
                for g, s in enumerate(uniq.tolist()):
                    row_idx = torch.where(inv == g)[0]              # batches with same start
                    rr = row_idx.repeat_interleave(num_heads)       # expand to BH rows
                    s = int(s)
                    # copy_ is much faster than scatter_ when writing contiguous blocks
                    k_dest_flat[rr, s:s+seq_new, :].copy_(k_src_flat[rr])
                    v_dest_flat[rr, s:s+seq_new, :].copy_(v_src_flat[rr])
        else:
            # Fallback: masked per-element write only when you truly need it
            time_valid = valid_token_mask_new.to(k_new.device, torch.bool)
            local_idx = time_valid.long().cumsum(1) - 1
            local_idx = local_idx.masked_fill(~time_valid, 0)
            target_idx = self.positions_offset[:, None] + local_idx                # [bs, seq_new]
            target_idx_flat = target_idx.repeat_interleave(num_heads, 0)           # [bh, seq_new]
            mask_flat = time_valid.repeat_interleave(num_heads, 0)                 # [bh, seq_new]

            # if time_valid_flat.any():
            row_ids = torch.arange(bh, device=k_new.device)[:, None].expand(bh, seq_new)
            rows = row_ids[mask_flat]
            cols = target_idx_flat[mask_flat]
            k_dest_flat[rows, cols] = k_src_flat[mask_flat]
            v_dest_flat[rows, cols] = v_src_flat[mask_flat]

        # Save back
        self.keys[layer_i] = k_dest_flat.view(bs, num_heads, self.cache_len, head_dim)
        self.values[layer_i] = v_dest_flat.view(bs, num_heads, self.cache_len, head_dim)


    def _layer_increase_size(self, layer_i, pad_len):
            assert 0 <= layer_i < self.num_layers
            k = self.keys[layer_i]
            v = self.values[layer_i]
            pad_shape = (k.size(0), k.size(1), pad_len, k.size(3))
            k_pad = torch.zeros(pad_shape, dtype=k.dtype, device=k.device)
            v_pad = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
            self.keys[layer_i] = torch.cat([k, k_pad], dim=2)
            self.values[layer_i] = torch.cat([v, v_pad], dim=2)


    def _ensure_capacity(self, needed_len: int) -> None:
        """Extend cached tensors along sequence dimension if capacity is insufficient."""
        if needed_len <= self.cache_len:
            return
        # potentially more than needed to create a buffer
        # new_cache_len = max(needed_len, int(self.cache_len * 1.5) + 256)
        new_cache_len = needed_len
        pad_len = new_cache_len - self.cache_len
        for i in range(self.num_layers):
            self._layer_increase_size(i, pad_len)
        self.cache_len = new_cache_len
    

    def get_active_cache_mask(self) -> Tensor:
        """Return [bs, cache_len] bool mask: True for active cached positions."""
        device = self.keys[0].device
        arange = torch.arange(self.cache_len, device=device)
        return arange[None, :] < self.positions_offset[:, None]

    @property
    def valid_token_mask(self) -> Tensor:
        """Compatibility: compute mask on-the-fly from positions_offset."""
        return self.get_active_cache_mask()



class EncoderDecoderCache:
    def __init__(
        self,
        self_attention_cache: Union[KVCache, List[Tuple[Tensor, Tensor]], None] = None,
        cross_attention_cache: Union[KVCache, List[Tuple[Tensor, Tensor]], None] = None,
        self_valid_token_mask: Optional[Tensor] = None,   # [bs, trg_time_dim]
        cross_valid_token_mask: Optional[Tensor] = None,  # [bs, src_time_dim]
        self_impl: Union[str, "KVCacheImpl"] = "pointer",  # NEW
    ):
        """
        Wrapper holding self- and cross-attention caches.
        """
        # print(f"{self_impl=}")
        # NEW: normalize and store impl for future cache (re)creation
        self.self_impl = KVCacheImpl(self_impl) if isinstance(self_impl, str) else self_impl

        # Self-attention cache (dynamic)
        if isinstance(self_attention_cache, MutableKVCache) or self_attention_cache is None:
            self.self_attention_cache = self_attention_cache
        elif isinstance(self_attention_cache, list):
            self.self_attention_cache = build_mutable_kv_cache(self.self_impl, self_attention_cache, valid_token_mask=self_valid_token_mask)
        else:
            raise TypeError("Invalid type for self_attention_cache")

        # Cross-attention cache (static)
        if isinstance(cross_attention_cache, StaticKVCache) or cross_attention_cache is None:
            self.cross_attention_cache = cross_attention_cache
        elif isinstance(cross_attention_cache, list):
            self.cross_attention_cache = StaticKVCache(cross_attention_cache, valid_token_mask=cross_valid_token_mask)
        else:
            raise TypeError("Invalid type for cross_attention_cache")


    def update_decoder_cache(
        self,
        new_self_kv: List[Tuple[Tensor, Tensor]] = None,
        new_cross_kv: List[Tuple[Tensor, Tensor]] = None,
        self_valid_token_mask_new: Optional[Tensor] = None,
        cross_valid_token_mask_new: Optional[Tensor] = None,
    ) -> None:
        """
        Update caches with new per-layer KV tuples.
        """
        # Update self-attention dynamic cache
        if new_self_kv is not None:
            if self.self_attention_cache is None:
                self.self_attention_cache = build_mutable_kv_cache(self.self_impl, new_self_kv, valid_token_mask=self_valid_token_mask_new)
            else:
                assert len(new_self_kv) == self.self_attention_cache.num_layers
                self.self_attention_cache.update_cache(new_self_kv, valid_token_mask_new=self_valid_token_mask_new)

        # Initialize cross-attention static cache once
        if new_cross_kv is not None and self.cross_attention_cache is None:
            self.cross_attention_cache = StaticKVCache(new_cross_kv, valid_token_mask=cross_valid_token_mask_new)

    def get_layer(self, layer_i):
        return self.get_self(layer_i), self.get_cross(layer_i)

    def get_self(self, layer_i): 
        if not self.self_attention_cache: return None
        return self.self_attention_cache.get_layer(layer_i)
    
    def get_cross(self, layer_i): 
        if not self.cross_attention_cache: return None
        return self.cross_attention_cache.get_layer(layer_i)
    
    def get_dynamic_active_cache_mask(self):
        return self.self_attention_cache.get_active_cache_mask()
    
    def get_dynamic_cache_positions(self):
        return self.self_attention_cache.cache_positions

    def get_positions_offset(self):
        return self.self_attention_cache.get_positions_offset()
    
    def has_self_attention_cache(self):
        return self.self_attention_cache is not None

    def has_cross_attention_cache(self):
        return self.cross_attention_cache is not None
    

    def index_select(self, index: Tensor):
        """
        Return a new EncoderDecoderCache reindexed along the batch dimension using `index`.
        """
        self_cache = self.self_attention_cache.index_select(index) if self.has_self_attention_cache() else None
        cross_cache = self.cross_attention_cache.index_select(index) if self.has_cross_attention_cache() else None

        # Derive corresponding valid masks if present
        self_valid = self_cache.valid_token_mask if self_cache is not None else None
        cross_valid = cross_cache.valid_token_mask if cross_cache is not None else None

        return EncoderDecoderCache(
            self_attention_cache=self_cache,
            cross_attention_cache=cross_cache,
            self_valid_token_mask=self_valid,
            cross_valid_token_mask=cross_valid,
            self_impl=self.self_impl,
        )
    


def build_mutable_kv_cache(
    impl: Union[str, "KVCacheImpl"],
    layers_kv: List[Tuple[Tensor, Tensor]],
    valid_token_mask: Optional[Tensor] = None,
) -> "MutableKVCache":
    if isinstance(impl, str):
        impl = KVCacheImpl(impl)
    if impl == KVCacheImpl.pointer:
        return KVCachePointer(layers_kv, valid_token_mask=valid_token_mask)
    if impl == KVCacheImpl.stack:
        return KVCacheStack(layers_kv, valid_token_mask=valid_token_mask)
    raise ValueError(f"Unknown kv cache impl: {impl}")