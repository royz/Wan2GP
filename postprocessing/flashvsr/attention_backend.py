from __future__ import annotations

import importlib
from typing import Callable

import torch


_SPARSE_ATTENTION: Callable | None = None
_BACKEND_NAME: str | None = None
_BACKEND_ERROR: str | None = None
_PRINTED_BACKEND = False


def _load_sparge() -> tuple[Callable | None, str | None]:
    for module_name in ("spas_sage_attn", "spas_sage_attn.core"):
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, "block_sparse_sage2_attn_cuda", None)
            if callable(fn):
                return fn, "SpargeAttn block_sparse_sage2_attn_cuda"
            fn = getattr(module, "spas_sage2_attn_meansim_topk_cuda", None)
            if callable(fn):
                return fn, "SpargeAttn spas_sage2_attn_meansim_topk_cuda"
            fn = getattr(module, "spas_sage2_attn_meansim_cuda", None)
            if callable(fn):
                return fn, "SpargeAttn spas_sage2_attn_meansim_cuda"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return None, locals().get("last_error", "dynamic SpargeAttn kernel not found")


def _mask_topk(mask_id: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor | float:
    if mask_id is None or not torch.is_tensor(mask_id):
        return 0.5
    density = mask_id.to(device=q.device, dtype=torch.float32).mean(dim=(0, 2, 3))
    return density.clamp(1.0 / max(int(mask_id.shape[-1]), 1), 1.0)


def _load_backend() -> tuple[Callable, str]:
    global _BACKEND_ERROR
    sparge_fn, sparge_name_or_error = _load_sparge()
    if sparge_fn is not None:
        def sparge_attention(qkv_list: list[torch.Tensor], mask_id: torch.Tensor) -> torch.Tensor:
            q, k, v = qkv_list
            qkv_list.clear()
            if "mask_id" in sparge_fn.__code__.co_varnames:
                return sparge_fn(q, k, v, mask_id=mask_id.to(torch.int8), tensor_layout="HND", output_dtype=q.dtype)
            if "topk" in sparge_fn.__code__.co_varnames:
                return sparge_fn(q, k, v, is_causal=False, tensor_layout="HND", output_dtype=q.dtype, topk=_mask_topk(mask_id, q))
            return sparge_fn(q, k, v, is_causal=False, tensor_layout="HND", output_dtype=q.dtype)

        return sparge_attention, sparge_name_or_error or "SpargeAttn"

    try:
        from .sparse_sage.core import sparse_sageattn

        def bundled_sparse_sage(qkv_list: list[torch.Tensor], mask_id: torch.Tensor) -> torch.Tensor:
            return sparse_sageattn(qkv_list, mask_id=mask_id.to(torch.int8), is_causal=False, tensor_layout="HND")

        return bundled_sparse_sage, "bundled Sparse SageAttention"
    except Exception as exc:
        bundled_error = f"{type(exc).__name__}: {exc}"

    _BACKEND_ERROR = f"SpargeAttn import failed: {sparge_name_or_error or 'not installed'}; bundled sparse-sage import failed: {bundled_error}"
    raise RuntimeError(
        "FlashVSR requires sparse attention. Install SpargeAttn (`spas_sage_attn`) or the bundled sparse-sage Triton fallback dependencies. "
        + _BACKEND_ERROR
    )


def get_sparse_backend_name() -> str:
    global _SPARSE_ATTENTION, _BACKEND_NAME
    if _SPARSE_ATTENTION is None:
        _SPARSE_ATTENTION, _BACKEND_NAME = _load_backend()
    return _BACKEND_NAME or "unknown"


def sparse_attention(qkv_list: list[torch.Tensor], mask_id: torch.Tensor) -> torch.Tensor:
    global _PRINTED_BACKEND
    backend_name = get_sparse_backend_name()
    if not _PRINTED_BACKEND:
        print(f"[FlashVSR] Sparse attention backend: {backend_name}")
        _PRINTED_BACKEND = True
    return _SPARSE_ATTENTION(qkv_list, mask_id)
