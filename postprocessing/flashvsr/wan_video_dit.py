import torch
import torch.nn as nn
import math
import random
import os
import time
from typing import Tuple, Optional, List
from einops import rearrange
from .utils import hash_state_dict_keys

try:
    from block_sparse_attn import block_sparse_attn_func
    BLOCK_ATTN_AVAILABLE = True
except:
    BLOCK_ATTN_AVAILABLE = False

from .attention_backend import sparse_attention
from shared.attention import get_supported_attention_modes, pay_attention
from mmgp import offload
from PIL import Image
import numpy as np

USE_BLOCK_ATTN = False
_FLASHVSR_ATTENTION_MODE = None


def get_flashvsr_attention_mode():
    global _FLASHVSR_ATTENTION_MODE
    selected = offload.shared_state.get("_attention")
    if selected not in (None, "auto"):
        return selected
    if _FLASHVSR_ATTENTION_MODE is None:
        modes = get_supported_attention_modes()
        _FLASHVSR_ATTENTION_MODE = "sage2" if "sage2" in modes else "sage" if "sage" in modes else "sdpa"
        print(f"[FlashVSR] WanGP attention backend: {_FLASHVSR_ATTENTION_MODE}")
    return _FLASHVSR_ATTENTION_MODE

# ----------------------------
# Local / window masks
# ----------------------------
@torch.no_grad()
def build_local_block_mask_shifted_vec(block_h: int,
                                       block_w: int,
                                       win_h: int = 6,
                                       win_w: int = 6,
                                       include_self: bool = True,
                                       device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = torch.clamp(r_all - r_half, 0, H - win_h)
    end_r   = start_r + win_h - 1
    start_c = torch.clamp(c_all - c_half, 0, W - win_w)
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask

@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                   block_w: int,
                                                   win_h: int = 6,
                                                   win_w: int = 6,
                                                   include_self: bool = True,
                                                   device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r   = start_r + win_h - 1
    start_c = c_all - c_half
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""
    @staticmethod
    def partition(x: torch.Tensor | list[torch.Tensor], win: Tuple[int, int, int]):
        if isinstance(x, list):
            x_list = x
            x = x_list[0]
            x_list.clear()
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor | list[torch.Tensor], win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        if isinstance(windows, list):
            windows_list = windows
            windows = windows_list[0]
            windows_list.clear()
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              qk_list, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    q_w, k_w = qk_list
    qk_list.clear()
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_k = torch.mean(k_w, dim=1)
    del q_w, k_w
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads)
    scores.mul_(1 / math.sqrt(D))

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    scores.masked_fill_(local_attn_mask.logical_not_(), -float('inf'))
    del local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    del scores
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1]-1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)  # keep shape note
    # 修正：上行变量名统一
    # mask_new = rearrange(attn_map, 'h (it s1) s2 -> h (it s1) s2', it=seqlen) * 0 + mask_new
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


@torch.no_grad()
def generate_draft_block_mask_sage(batch_size, nheads, seqlen,
                                      qk_list, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    
    q_w, k_w = qk_list
    qk_list.clear()
    avgpool_q = torch.mean(q_w, dim=1) 
    del q_w
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    
    k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
    avgpool_k_split = torch.mean(k_w_split, dim=2)
    del k_w, k_w_split
    avgpool_k_refined = rearrange(avgpool_k_split, 's two d -> (s two) d', two=2) # shape: (s*2, C)
    avgpool_k_refined = rearrange(avgpool_k_refined, 's (h d) -> s h d', h=nheads) # shape: (s*2, h, d)
    k_heads_doubled = avgpool_k_refined.permute(1, 0, 2) # shape: (h, s*2, d)
    
    k_heads_1, k_heads_2 = torch.chunk(k_heads_doubled, 2, dim=1)
    scores_1 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_1)
    scores_1.mul_(1 / math.sqrt(D))
    scores_2 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_2)
    scores_2.mul_(1 / math.sqrt(D))
    scores = torch.cat([scores_1, scores_2], dim=-1)
    del scores_1, scores_2

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = (scores.shape[2] // 2) // local_attn_mask.shape[1]
    
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.repeat_interleave(2, dim=1)
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    
    assert scores.shape == local_attn_mask.shape, \
        f"Scores shape {scores.shape} != Mask shape {local_attn_mask.shape}"
    
    scores.masked_fill_(local_attn_mask.logical_not_(), -float('inf'))
    del local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    del scores
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1]-1, topk)
    
    if apply_topk <= 0:
        mask_new = torch.zeros_like(flat, dtype=torch.bool).reshape(loop_num, s1, s2)
    else:
        thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
        thresholds = thresholds.unsqueeze(1)
        mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
        
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


# ----------------------------
# Attention kernels
# ----------------------------
def flash_attention(qkv_list: list[torch.Tensor], num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False):
    q, k, v = qkv_list
    qkv_list.clear()
    if attention_mask is not None:
        seqlen = q.shape[1]
        seqlen_kv = k.shape[1]
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
        head_mask_type = torch.tensor([1]*num_heads, device=q.device, dtype=torch.int32)
        streaming_info = None
        base_blockmask = attention_mask
        max_seqlen_q_ = seqlen
        max_seqlen_k_ = seqlen_kv
        p_dropout = 0.0
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            x = block_sparse_attn_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                deterministic=False,
                softmax_scale=None,
                is_causal=False,
                exact_streaming=False,
                return_attn_probs=False,
            ).unsqueeze(0)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            x = sparse_attention([q, k, v], base_blockmask)
            q = k = v = None
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = pay_attention([q, k, v], force_attention="sdpa" if compatibility_mode else get_flashvsr_attention_mode(), recycle_q=True)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x.mul_(scale.add_(1)).add_(shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def _rope_axis_inplace(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    scratch = x_even.clone()
    x_even.mul_(cos).addcmul_(x_odd, sin, value=-1)
    x_odd.mul_(cos).addcmul_(scratch, sin)
    del scratch


def rope_apply(x_list, freqs, num_heads, f: int, h: int, w: int):
    x = x_list[0]
    x_list.clear()
    b = x.shape[0]
    x = x.view(b, f, h, w, num_heads, -1)
    f_freqs, h_freqs, w_freqs = freqs
    f_dim = f_freqs[0].shape[-1] * 2
    h_dim = h_freqs[0].shape[-1] * 2
    _rope_axis_inplace(x[..., :f_dim], f_freqs[0].view(1, f, 1, 1, 1, -1), f_freqs[1].view(1, f, 1, 1, 1, -1))
    _rope_axis_inplace(x[..., f_dim:f_dim + h_dim], h_freqs[0].view(1, 1, h, 1, 1, -1), h_freqs[1].view(1, 1, h, 1, 1, -1))
    _rope_axis_inplace(x[..., f_dim + h_dim:], w_freqs[0].view(1, 1, 1, w, 1, -1), w_freqs[1].view(1, 1, 1, w, 1, -1))
    return x.reshape(b, f * h * w, -1)


# ----------------------------
# Norms & Blocks
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype).mul_(self.weight)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, qkv_list, attention_mask=None):
        x = flash_attention(qkv_list, num_heads=self.num_heads, attention_mask=attention_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(self, x, freqs, f=None, h=None, w=None, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, pre_cache_refs=None, local_range = 9, cache_next=True):
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        if pre_cache_refs is not None:
            pre_cache_k, pre_cache_v = pre_cache_refs
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f==2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f==6, " start f must be 6"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        del x
        q_list = [q]
        q = None
        q = rope_apply(q_list, freqs, self.num_heads, f, h, w)
        k_list = [k]
        k = None
        k = rope_apply(k_list, freqs, self.num_heads, f, h, w)

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)
        k = k.view(B, f, h, w, D)
        v = v.view(B, f, h, w, D)

        q_w = WindowPartition3D.partition([q], win)
        q = None
        k_w = WindowPartition3D.partition([k], win)
        k = None
        v_w = WindowPartition3D.partition([v], win)
        v = None

        seqlen = f//win[0]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)
            if pre_cache_refs is not None:
                pre_cache_refs[0] = None
                pre_cache_refs[1] = None
            pre_cache_k = None
            pre_cache_v = None

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n, block_s=block_s)
        reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)
        reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)

        window_size = win[0]*h*w//128

        if self.local_attn_mask is None or self.local_attn_mask_h!=h//8 or self.local_attn_mask_w!=w//8 or self.local_range!=local_range:
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(h//8, w//8, local_range, local_range, include_self=True, device=k_w.device)
            self.local_attn_mask_h = h//8
            self.local_attn_mask_w = w//8
            self.local_range = local_range
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            attention_mask = generate_draft_block_mask(B, self.num_heads, seqlen, [q_w, k_w], topk=topk, local_attn_mask=self.local_attn_mask)
            q_w = None
        else:
            attention_mask = generate_draft_block_mask_sage(B, self.num_heads, seqlen, [q_w, k_w], topk=topk, local_attn_mask=self.local_attn_mask)
            q_w = None

        x = self.attn([reorder_q, reorder_k, reorder_v], attention_mask)
        reorder_q = reorder_k = reorder_v = None
        del attention_mask

        cache_k = cache_v = None
        if is_stream and cache_next:
            cache_blocks = min(block_n_kv, one_len * max(1, int(kv_len)))
            cache_k = k_w.view(B, block_n_kv, block_s, D)[:, -cache_blocks:].reshape(B * cache_blocks, block_s, D)
            cache_v = v_w.view(B, block_n_kv, block_s, D)[:, -cache_blocks:].reshape(B * cache_blocks, block_s, D)
            cache_k = cache_k.detach().to("cpu")
            cache_v = cache_v.detach().to("cpu")
        if is_stream:
            del k_w, v_w

        x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d', block_n=block_n, block_s=block_s)
        x = WindowPartition3D.reverse([x], win, (f, h, w))
        x = x.view(B, f*h*w, D)

        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)


class CrossAttention(nn.Module):
    """
    仅考虑文本 context；提供持久 KV 缓存。
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

        # 持久缓存
        self.cache_k = None
        self.cache_v = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        """ctx: [B, S_ctx, dim] —— 经过 text_embedding 之后的上下文"""
        self.cache_k = self.norm_k(self.k(ctx))
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor | list[torch.Tensor], y: torch.Tensor, is_stream: bool = False):
        """
        y 即文本上下文（未做其他分支）。
        """
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        q = self.norm_q(self.q(x))
        del x
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v

        x = self.attn([q, k, v])
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x.add_(residual.mul_(gate))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = max(1, min(x.shape[1], int(x.shape[1] * self.dim / self.ffn_dim)))
        if chunk_size >= x.shape[1]:
            return self.ffn(x)
        if self.training:
            return torch.cat([self.ffn(chunk) for chunk in x.split(chunk_size, dim=1)], dim=1)
        out = torch.empty_like(x)
        for start in range(0, x.shape[1], chunk_size):
            end = min(start + chunk_size, x.shape[1])
            out[:, start:end] = self.ffn(x[:, start:end])
        return out

    def forward(self, x, context, t_mod, freqs, f, h, w, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, pre_cache_refs=None, local_range = 9, cache_next=True):
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_list = [input_x] 
        input_x = None
        self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
            x_list , freqs, f, h, w, local_num, topk, train_img, block_id,
            kv_len=kv_len, is_full_block=is_full_block, is_stream=is_stream,
            pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v, pre_cache_refs=pre_cache_refs, local_range = local_range, cache_next=cache_next)

        x = self.gate(x, gate_msa, self_attn_output)
        del self_attn_output
        cross_input = self.norm3(x)
        cross = self.cross_attn([cross_input], context, is_stream=is_stream)
        x.add_(cross)
        del cross, cross_input
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        ffn_out = self.forward_ffn(input_x)
        del input_x
        x = self.gate(x, gate_mlp, ffn_out)
        del ffn_out
        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x.add_(self.emb_pos.to(dtype=x.dtype, device=x.device))
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.head(x)


# ----------------------------
# WanModel (no image branch) — init 时即产生 KV 缓存
# ----------------------------
class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        # init_context: torch.Tensor,     # <<<< 必填：在 __init__ 里用它生成 cross-attn KV 缓存
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        # patch embed
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # text / time embed
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self._cross_kv_initialized = False

    # 可选:手动清空 / 重新初始化
    def clear_cross_kv(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    @torch.no_grad()
    def reinit_cross_kv(self, new_context: torch.Tensor):
        ctx_txt = self.text_embedding(new_context)
        for blk in self.blocks:
            blk.cross_attn.init_cache(ctx_txt)
        self._cross_kv_initialized = True

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor | list[torch.Tensor], grid_size: torch.Tensor):
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                LQ_latents: Optional[List[torch.Tensor]] = None,
                train_img: bool = False,
                topk_ratio: Optional[float] = None,
                kv_ratio: Optional[float] = None,
                local_num: Optional[int] = None,
                is_full_block: bool = False,
                causal_idx: Optional[int] = None,
                **kwargs,
                ):
        # time / text embeds
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 这里仍会嵌入 text（CrossAttention 若已有缓存会忽略它）
        # context = self.text_embedding(context)

        # 输入打补丁
        x, (f, h, w) = self.patchify(x)
        B = x.shape[0]

        # window / masks 超参
        win = (2, 8, 8)
        seqlen = f//win[0]
        if local_num is None:
            local_random = random.random()
            if local_random < 0.3:
                local_num = seqlen - 3
            elif local_random < 0.4:
                local_num = seqlen - 4
            elif local_random < 0.5:
                local_num = seqlen - 2
            else:
                local_num = seqlen

        window_size = win[0]*h*w//128
        square_num = window_size*window_size
        topk_ratio = 2.0
        topk = min(max(int(square_num*topk_ratio), 1), int(square_num*seqlen)-1)

        if kv_ratio is None:
            kv_ratio = (random.uniform(0., 1.0)**2)*(local_num-2-2)+2
        kv_len = min(max(int(window_size*kv_ratio), 1), int(window_size*seqlen)-1)

        decay_ratio = random.uniform(0.7, 1.0)

        freqs = tuple((freq.real.to(device=x.device, dtype=x.dtype), freq.imag.to(device=x.device, dtype=x.dtype)) for freq in (self.freqs[0][:f], self.freqs[1][:h], self.freqs[2][:w]))

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # blocks
        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x.add_(LQ_latents[block_id])

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, f, h, w, local_num, topk,
                            train_img, block_id, kv_len, is_full_block, False,
                            None, None,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, f, h, w, local_num, topk,
                        train_img, block_id, kv_len, is_full_block, False,
                        None, None, 
                        use_reentrant=False,
                    )
            else:
                x = block([x], context, t_mod, freqs, f, h, w, local_num, topk,
                          train_img, block_id, kv_len, is_full_block, False,
                          None, None)

        x = self.head([x], t)
        x = self.unpatchify([x], (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    

# ----------------------------
# State dict converter（保持原映射；已忽略 has_image_input 使用）
# ----------------------------
class WanModelStateDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        # 保留原有哈希匹配返回的 config；实现本身不使用 has_image_input 分支
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6,"has_image_pos_emb": False}
        else:
            config = {}
        return state_dict, config
