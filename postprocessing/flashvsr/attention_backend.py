from __future__ import annotations

import importlib
from typing import Callable

import torch


_SPARSE_ATTENTION: Callable | None = None
_BACKEND_NAME: str | None = None
_BACKEND_ERROR: str | None = None
_PRINTED_BACKEND = False


def _load_sparge() -> tuple[Callable | None, str | None]:
    try:
        module = importlib.import_module("shared.spas_sage_attn_core")
        fn = getattr(module, "block_sparse_sage2_attn_cuda", None)
        if callable(fn):
            return fn, "WanGP SpargeAttn block_sparse_sage2_attn_cuda"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return None, "WanGP SpargeAttn block_sparse_sage2_attn_cuda not found"


def sparge_attention_available() -> bool:
    sparge_fn, _ = _load_sparge()
    return sparge_fn is not None


def require_sparge_attention() -> None:
    sparge_fn, sparge_error = _load_sparge()
    if sparge_fn is None:
        raise RuntimeError(f"FlashVSR requires SpargeAttn (`spas_sage_attn`). Install the SpargeAttn kernels from docs/INSTALLATION.md and restart WanGP. SpargeAttn import failed: {sparge_error or 'not installed'}")


def _mask_topk(mask_id: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor | float:
    if isinstance(mask_id, list):
        mask_id = mask_id[0] if len(mask_id) > 0 else None
    if mask_id is None or not torch.is_tensor(mask_id):
        return 0.5
    density = mask_id.to(device=q.device, dtype=torch.float32).mean(dim=(0, 2, 3))
    return density.clamp(1.0 / max(int(mask_id.shape[-1]), 1), 1.0)


def _int8_mask(mask_id: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    mask = mask_id[0] if isinstance(mask_id, list) else mask_id
    if mask.dtype != torch.int8:
        mask = mask.to(torch.int8)
        if isinstance(mask_id, list):
            mask_id[0] = mask
    return mask_id if isinstance(mask_id, list) else mask


def _take_mask(mask_id: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(mask_id, list):
        mask = mask_id[0]
        mask_id.clear()
        return mask
    return mask_id


def _load_backend() -> tuple[Callable, str]:
    global _BACKEND_ERROR
    sparge_fn, sparge_name_or_error = _load_sparge()
    if sparge_fn is not None:
        use_qkv_list = sparge_fn.__module__ == "shared.spas_sage_attn_core"

        def sparge_attention(qkv_list: list[torch.Tensor], mask_id: torch.Tensor | list[torch.Tensor], recycle_q: bool = False) -> torch.Tensor:
            if "mask_id" in sparge_fn.__code__.co_varnames:
                mask_id = _int8_mask(mask_id)
                if use_qkv_list:
                    return sparge_fn(qkv_list, mask_id=mask_id, tensor_layout="HND", output_dtype=qkv_list[0].dtype, recycle_q=recycle_q)
                q, k, v = qkv_list
                qkv_list.clear()
                return sparge_fn(q, k, v, mask_id=_take_mask(mask_id), tensor_layout="HND", output_dtype=q.dtype)
            if "topk" in sparge_fn.__code__.co_varnames:
                if use_qkv_list:
                    topk = _mask_topk(mask_id, qkv_list[0])
                    if isinstance(mask_id, list):
                        mask_id.clear()
                    return sparge_fn(qkv_list, is_causal=False, tensor_layout="HND", output_dtype=qkv_list[0].dtype, topk=topk, recycle_q=recycle_q)
                q, k, v = qkv_list
                qkv_list.clear()
                topk = _mask_topk(mask_id, q)
                if isinstance(mask_id, list):
                    mask_id.clear()
                return sparge_fn(q, k, v, is_causal=False, tensor_layout="HND", output_dtype=q.dtype, topk=topk)
            q, k, v = qkv_list
            qkv_list.clear()
            return sparge_fn(q, k, v, is_causal=False, tensor_layout="HND", output_dtype=q.dtype)

        return sparge_attention, sparge_name_or_error or "SpargeAttn"

    _BACKEND_ERROR = f"SpargeAttn import failed: {sparge_name_or_error or 'not installed'}"
    raise RuntimeError(f"FlashVSR requires SpargeAttn (`spas_sage_attn`). Install the SpargeAttn kernels from docs/INSTALLATION.md and restart WanGP. {_BACKEND_ERROR}")


def get_sparse_backend_name() -> str:
    global _SPARSE_ATTENTION, _BACKEND_NAME
    if _SPARSE_ATTENTION is None:
        _SPARSE_ATTENTION, _BACKEND_NAME = _load_backend()
    return _BACKEND_NAME or "unknown"


def sparse_attention(qkv_list: list[torch.Tensor], mask_id: torch.Tensor | list[torch.Tensor], recycle_q: bool = False) -> torch.Tensor:
    global _PRINTED_BACKEND
    backend_name = get_sparse_backend_name()
    if not _PRINTED_BACKEND:
        print(f"[FlashVSR] Sparse attention backend: {backend_name}")
        _PRINTED_BACKEND = True
    return _SPARSE_ATTENTION(qkv_list, mask_id, recycle_q=recycle_q)
