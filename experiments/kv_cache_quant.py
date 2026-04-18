"""
KV Cache Quantization Module for TurboQuant.

Provides hooks to intercept and quantize the KV cache of HuggingFace
transformer models on-the-fly during generation.

Supports mixed-precision: outlier channels get higher bit-width, regular
channels get lower bit-width, yielding non-integer effective bit-widths
(e.g., 2.5-bit, 3.5-bit) as described in Section 4.3 of the paper.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from turboquant import TurboQuantProd, TurboQuantMSE


class TurboQuantKVCache:
    """
    Wraps TurboQuant to quantize KV cache tensors on the fly.

    Supports mixed-precision with outlier channel detection:
      - Outlier channels (high magnitude) get `outlier_bits` bits
      - Regular channels get `regular_bits` bits
      - Effective bit-width = (n_outlier * outlier_bits + n_regular * regular_bits) / total_channels

    Example:
      2.5-bit: 32 outlier channels at 3 bits, 96 regular at 2 bits → (32*3 + 96*2)/128 = 2.5
      3.5-bit: 32 outlier channels at 4 bits, 96 regular at 3 bits → (32*4 + 96*3)/128 = 3.5
    """

    def __init__(
        self,
        head_dim: int,
        n_outlier_channels: int = 32,
        outlier_bits: int = 3,
        regular_bits: int = 2,
        device: torch.device = None,
        seed: int = 42,
        quantizer_type: str = 'prod',  # 'prod' or 'mse'
    ):
        self.head_dim = head_dim
        self.n_outlier = n_outlier_channels
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits
        self.device = device or torch.device('cpu')
        self.quantizer_type = quantizer_type

        QuantClass = TurboQuantProd if quantizer_type == 'prod' else TurboQuantMSE

        # Create quantizers for outlier and regular channels
        if n_outlier_channels > 0 and outlier_bits > 0:
            self.outlier_quantizer = QuantClass(
                n_outlier_channels, outlier_bits, device=self.device, seed=seed
            )
        else:
            self.outlier_quantizer = None

        n_regular = head_dim - n_outlier_channels
        if n_regular > 0 and regular_bits > 0:
            self.regular_quantizer = QuantClass(
                n_regular, regular_bits, device=self.device, seed=seed + 1
            )
        else:
            self.regular_quantizer = None

        self.effective_bits = (
            n_outlier_channels * outlier_bits + (head_dim - n_outlier_channels) * regular_bits
        ) / head_dim

    def move_to(self, device: torch.device):
        """Move all quantizer tensors to target device (lazy, called on first use)."""
        if self.device == device:
            return
        self.device = device
        for q in [self.outlier_quantizer, self.regular_quantizer]:
            if q is None:
                continue
            _move_quantizer(q, device)

    def quantize_dequantize(self, kv: torch.Tensor, outlier_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Quantize and dequantize KV cache tensor.

        Args:
            kv: (..., head_dim) tensor (any leading batch dims)
            outlier_indices: (n_outlier,) indices of outlier channels.
                If None, uses the first n_outlier channels.

        Returns:
            kv_hat: quantized-dequantized tensor, same shape as kv
        """
        orig_shape = kv.shape
        head_dim = orig_shape[-1]
        flat = kv.reshape(-1, head_dim).float()

        if outlier_indices is None:
            outlier_indices = torch.arange(self.n_outlier, device=self.device)

        regular_mask = torch.ones(head_dim, dtype=torch.bool, device=self.device)
        regular_mask[outlier_indices] = False
        regular_indices = torch.where(regular_mask)[0]

        result = torch.zeros_like(flat)

        # Quantize outlier channels
        if self.outlier_quantizer is not None and len(outlier_indices) > 0:
            outlier_data = flat[:, outlier_indices]
            outlier_hat = self.outlier_quantizer.quantize_dequantize(outlier_data)
            result[:, outlier_indices] = outlier_hat

        # Quantize regular channels
        if self.regular_quantizer is not None and len(regular_indices) > 0:
            regular_data = flat[:, regular_indices]
            regular_hat = self.regular_quantizer.quantize_dequantize(regular_data)
            result[:, regular_indices] = regular_hat

        return result.reshape(orig_shape).to(kv.dtype)


def detect_outlier_channels(kv_tensor: torch.Tensor, n_outlier: int) -> torch.Tensor:
    """
    Detect outlier channels based on magnitude.
    Uses the mean absolute value per channel across all tokens.

    Args:
        kv_tensor: (batch, n_heads, seq_len, head_dim) or (seq_len, head_dim)
        n_outlier: number of outlier channels to select

    Returns:
        outlier_indices: (n_outlier,) tensor of channel indices
    """
    if kv_tensor.dim() == 4:
        # (batch, n_heads, seq_len, head_dim) -> mean over batch, heads, seq
        magnitudes = kv_tensor.abs().mean(dim=(0, 1, 2))
    elif kv_tensor.dim() == 3:
        magnitudes = kv_tensor.abs().mean(dim=(0, 1))
    elif kv_tensor.dim() == 2:
        magnitudes = kv_tensor.abs().mean(dim=0)
    else:
        magnitudes = kv_tensor.abs()

    _, outlier_indices = magnitudes.topk(n_outlier)
    return outlier_indices.sort()[0]


def _move_quantizer(q, device):
    """Recursively move all tensor attributes of a TurboQuant quantizer to device."""
    for attr in ['Pi', 'centroids']:
        if hasattr(q, attr):
            setattr(q, attr, getattr(q, attr).to(device))
    if hasattr(q, 'device'):
        q.device = device
    if hasattr(q, 'qjl') and q.qjl is not None:
        if hasattr(q.qjl, 'S'):
            q.qjl.S = q.qjl.S.to(device)
        q.qjl.device = device
    if hasattr(q, 'mse_quantizer') and q.mse_quantizer is not None:
        _move_quantizer(q.mse_quantizer, device)


def apply_turboquant_to_kv_cache(
    model,
    effective_bits: float = 2.5,
    n_outlier_channels: int = 32,
    quantizer_type: str = 'prod',
    device: torch.device = None,
    seed: int = 42,
):
    """
    Monkey-patch a HuggingFace model to apply TurboQuant on its KV cache.

    This hooks into the model's cache update mechanism to quantize
    key and value states after they are computed.

    Args:
        model: HuggingFace model (e.g., LlamaForCausalLM)
        effective_bits: target effective bit-width (e.g., 2.5, 3.5)
        n_outlier_channels: number of outlier channels per head
        quantizer_type: 'prod' or 'mse'
        device: torch device
        seed: random seed
    """
    config = model.config
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

    # Compute outlier_bits and regular_bits from effective_bits
    n_regular = head_dim - n_outlier_channels
    # effective_bits = (n_outlier * outlier_bits + n_regular * regular_bits) / head_dim
    # Typically: outlier_bits = regular_bits + 1
    regular_bits = int(math.floor(effective_bits))
    outlier_bits = regular_bits + 1
    # Verify: check if it matches, else adjust
    actual_eff = (n_outlier_channels * outlier_bits + n_regular * regular_bits) / head_dim
    if abs(actual_eff - effective_bits) > 0.01:
        # Try other combinations
        for rb in range(1, 8):
            for ob in range(rb, rb + 3):
                eff = (n_outlier_channels * ob + n_regular * rb) / head_dim
                if abs(eff - effective_bits) < 0.01:
                    regular_bits, outlier_bits = rb, ob
                    break

    print(f"TurboQuant KV cache: head_dim={head_dim}, "
          f"outlier={n_outlier_channels}ch@{outlier_bits}bit, "
          f"regular={n_regular}ch@{regular_bits}bit, "
          f"effective={effective_bits:.1f}bit")

    # Create per-layer quantizers for keys and values
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)

    kv_quantizers = {}
    for layer_idx in range(n_layers):
        for kv in ['k', 'v']:
            kv_quantizers[(layer_idx, kv)] = TurboQuantKVCache(
                head_dim=head_dim,
                n_outlier_channels=n_outlier_channels,
                outlier_bits=outlier_bits,
                regular_bits=regular_bits,
                device=device,
                seed=seed + layer_idx * 2 + (0 if kv == 'k' else 1),
                quantizer_type=quantizer_type,
            )

    # Store quantizers on model for access
    model._tq_kv_quantizers = kv_quantizers
    model._tq_outlier_indices = {}  # will be populated per layer

    # Hook into each attention layer
    _patch_attention_layers(model, kv_quantizers)

    return model


def _patch_attention_layers(model, kv_quantizers):
    """
    Patch the model's KV cache update method to quantize key/value states
    before they are stored.

    Works across transformers versions:
      - transformers >= 5.0: DynamicCache.update(key, value, layer_idx)
      - transformers <  5.0: per-layer attention forward hook
    """
    try:
        from transformers.cache_utils import DynamicCache
        _patch_dynamic_cache(DynamicCache, kv_quantizers)
        print("TurboQuant: patched DynamicCache.update (transformers >= 5.0 path)")
    except Exception as e:
        print(f"TurboQuant: DynamicCache patch failed ({e}), falling back to attention hooks")
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            _patch_single_attention(attn, layer_idx, kv_quantizers)


def _patch_dynamic_cache(DynamicCache, kv_quantizers):
    """
    Patch DynamicCache.update to quantize KV states on write.
    This is the correct hook point for transformers >= 5.0.
    """
    original_update = DynamicCache.update

    def quantized_update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if (layer_idx, 'k') in kv_quantizers:
            k_q = kv_quantizers[(layer_idx, 'k')]
            v_q = kv_quantizers[(layer_idx, 'v')]
            # Move quantizer codebooks to match key_states device on first use
            dev = key_states.device
            k_q.move_to(dev)
            v_q.move_to(dev)
            # key_states shape: (batch, n_heads, seq_len, head_dim)
            orig_k_shape = key_states.shape
            orig_v_shape = value_states.shape
            key_states   = k_q.quantize_dequantize(key_states.reshape(-1, orig_k_shape[-1])).reshape(orig_k_shape)
            value_states = v_q.quantize_dequantize(value_states.reshape(-1, orig_v_shape[-1])).reshape(orig_v_shape)
        return original_update(self, key_states, value_states, layer_idx, *args, **kwargs)

    DynamicCache.update = quantized_update


def _patch_single_attention(attn_module, layer_idx, kv_quantizers):
    """
    Patch a single attention module to quantize its KV states.
    We wrap the attention forward to quantize the KV cache after it's updated.
    """
    original_forward = attn_module.forward

    def quantized_forward(*args, **kwargs):
        # Force eager attention to avoid fused kernels bypassing Python hooks
        if 'attn_implementation' not in kwargs:
            pass  # attn_implementation is set at model level, not per-call

        outputs = original_forward(*args, **kwargs)

        # Find the Cache object in the outputs tuple.
        # Different HF versions return different tuple lengths:
        #   - (attn_output, attn_weights, past_kv)  — len 3
        #   - (attn_output, past_kv)                 — len 2 (when output_attentions=False)
        if isinstance(outputs, tuple):
            past_kv = None
            for item in outputs:
                if hasattr(item, 'layers') or hasattr(item, 'key_cache') or hasattr(item, 'value_cache'):
                    past_kv = item
                    break

            if past_kv is not None:
                k_q = kv_quantizers[(layer_idx, 'k')]
                v_q = kv_quantizers[(layer_idx, 'v')]

                # transformers >= 5.0: DynamicCache uses .layers[i].keys / .values
                if hasattr(past_kv, 'layers') and layer_idx < len(past_kv.layers):
                    layer_cache = past_kv.layers[layer_idx]
                    k = layer_cache.keys   # (batch, n_heads, seq_len, head_dim)
                    v = layer_cache.values
                    k_flat = k.reshape(-1, k.shape[-1])
                    v_flat = v.reshape(-1, v.shape[-1])
                    layer_cache.keys   = k_q.quantize_dequantize(k_flat).reshape(k.shape)
                    layer_cache.values = v_q.quantize_dequantize(v_flat).reshape(v.shape)

                # transformers < 5.0: DynamicCache uses .key_cache / .value_cache lists
                elif hasattr(past_kv, 'key_cache') and layer_idx < len(past_kv.key_cache):
                    k = past_kv.key_cache[layer_idx]
                    v = past_kv.value_cache[layer_idx]
                    k_flat = k.reshape(-1, k.shape[-1])
                    v_flat = v.reshape(-1, v.shape[-1])
                    past_kv.key_cache[layer_idx]   = k_q.quantize_dequantize(k_flat).reshape(k.shape)
                    past_kv.value_cache[layer_idx] = v_q.quantize_dequantize(v_flat).reshape(v.shape)

        return outputs

    attn_module.forward = quantized_forward
