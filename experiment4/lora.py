from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


LAYER_PATTERNS = (
    re.compile(r"model\.layers\.(\d+)\."),      # LLaMA/OLMo/Mistral style
    re.compile(r"transformer\.h\.(\d+)\."),     # GPT-2 style
)
ATTN_PROJ_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "attn.c_attn",  # GPT-2 fused QKV
    "attn.c_proj",  # GPT-2 attention output
)
MLP_PROJ_SUFFIXES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.c_fc",     # GPT-2 MLP input
    "mlp.c_proj",   # GPT-2 MLP output
)
ALL_SUFFIXES = ATTN_PROJ_SUFFIXES + MLP_PROJ_SUFFIXES


@dataclass(frozen=True)
class LoRAHyperParams:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    si_rank: int = 1
    # Condition 'e' (SI-amplified): SI heads get si_amplified_rank,
    # non-SI heads get non_si_reduced_rank.  Defaults chosen so total
    # trainable parameter budget roughly matches condition 'd' (rank=8
    # everywhere): 16 on ~25% of heads + 4 on ~75% ≈ 7.0 average rank.
    si_amplified_rank: int = 16
    non_si_reduced_rank: int = 4


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        in_mask: torch.Tensor | None = None,
        out_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRALinear rank must be > 0")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(max(1, rank))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        if hasattr(base, "in_features") and hasattr(base, "out_features"):
            in_features = int(getattr(base, "in_features"))
            out_features = int(getattr(base, "out_features"))
        else:
            # transformers Conv1D-style modules expose a [in, out] weight.
            w = getattr(base, "weight", None)
            if w is None or int(getattr(w, "ndim", 0)) != 2:
                raise TypeError(f"Unsupported base module for LoRA: {type(base).__name__}")
            in_features = int(w.shape[0])
            out_features = int(w.shape[1])
        self.lora_a = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

        if in_mask is None:
            self.register_buffer("in_mask", None)
        else:
            self.register_buffer("in_mask", in_mask.reshape(1, -1).to(dtype=torch.float32))
        if out_mask is None:
            self.register_buffer("out_mask", None)
        else:
            self.register_buffer("out_mask", out_mask.reshape(1, -1).to(dtype=torch.float32))

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_x = x
        if self.in_mask is not None:
            lora_x = lora_x * self.in_mask.to(device=lora_x.device, dtype=lora_x.dtype)
        update = F.linear(self.dropout(lora_x), self.lora_a)
        update = F.linear(update, self.lora_b)
        if self.out_mask is not None:
            update = update * self.out_mask.to(device=update.device, dtype=update.dtype)
        update = update * self.scaling
        return base_out + update.to(dtype=base_out.dtype)


class CompositeLoRALinear(nn.Module):
    """Two LoRA adapters on the same base Linear with complementary masks.

    Used by conditions 'a' and 'e' for attention projections in layers that
    contain SI heads:
      - Condition 'a' (SI-protecting): rank_main=full on non-SI, rank_si=1 on SI.
      - Condition 'e' (SI-amplified): rank_main=reduced on non-SI, rank_si=high on SI.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank_main: int,
        rank_si: int,
        alpha: float,
        dropout: float,
        in_mask_main: torch.Tensor | None,
        out_mask_main: torch.Tensor | None,
        in_mask_si: torch.Tensor | None,
        out_mask_si: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = int(base.in_features)
        out_features = int(base.out_features)

        # Main adapter (full rank, non-SI heads)
        self.rank_main = int(rank_main)
        self.scaling_main = float(alpha) / float(max(1, rank_main))
        self.dropout_main = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a_main = nn.Parameter(torch.zeros(self.rank_main, in_features))
        self.lora_b_main = nn.Parameter(torch.zeros(out_features, self.rank_main))
        nn.init.kaiming_uniform_(self.lora_a_main, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_main)

        if in_mask_main is None:
            self.register_buffer("in_mask_main", None)
        else:
            self.register_buffer("in_mask_main", in_mask_main.reshape(1, -1).to(dtype=torch.float32))
        if out_mask_main is None:
            self.register_buffer("out_mask_main", None)
        else:
            self.register_buffer("out_mask_main", out_mask_main.reshape(1, -1).to(dtype=torch.float32))

        # SI adapter (low rank, SI heads)
        self.rank_si = int(rank_si)
        self.scaling_si = float(alpha) / float(max(1, rank_si))
        self.dropout_si = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a_si = nn.Parameter(torch.zeros(self.rank_si, in_features))
        self.lora_b_si = nn.Parameter(torch.zeros(out_features, self.rank_si))
        nn.init.kaiming_uniform_(self.lora_a_si, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_si)

        if in_mask_si is None:
            self.register_buffer("in_mask_si", None)
        else:
            self.register_buffer("in_mask_si", in_mask_si.reshape(1, -1).to(dtype=torch.float32))
        if out_mask_si is None:
            self.register_buffer("out_mask_si", None)
        else:
            self.register_buffer("out_mask_si", out_mask_si.reshape(1, -1).to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        # Main (non-SI) adapter
        lora_x = x
        if self.in_mask_main is not None:
            lora_x = lora_x * self.in_mask_main.to(device=lora_x.device, dtype=lora_x.dtype)
        update_main = F.linear(self.dropout_main(lora_x), self.lora_a_main)
        update_main = F.linear(update_main, self.lora_b_main)
        if self.out_mask_main is not None:
            update_main = update_main * self.out_mask_main.to(device=update_main.device, dtype=update_main.dtype)
        update_main = update_main * self.scaling_main

        # SI adapter
        lora_x_si = x
        if self.in_mask_si is not None:
            lora_x_si = lora_x_si * self.in_mask_si.to(device=lora_x_si.device, dtype=lora_x_si.dtype)
        update_si = F.linear(self.dropout_si(lora_x_si), self.lora_a_si)
        update_si = F.linear(update_si, self.lora_b_si)
        if self.out_mask_si is not None:
            update_si = update_si * self.out_mask_si.to(device=update_si.device, dtype=update_si.dtype)
        update_si = update_si * self.scaling_si

        combined = update_main.to(dtype=base_out.dtype) + update_si.to(dtype=base_out.dtype)
        return base_out + combined


@dataclass
class LoRAApplyResult:
    replaced_modules: list[str]
    trainable_params: int
    total_params: int


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _parse_layer_idx(module_name: str) -> int | None:
    for pat in LAYER_PATTERNS:
        m = pat.search(module_name)
        if m:
            return int(m.group(1))
    return None


def _split_parent_child(module_name: str) -> tuple[str, str]:
    if "." not in module_name:
        return "", module_name
    parent, child = module_name.rsplit(".", 1)
    return parent, child


def _get_module(root: nn.Module, module_name: str) -> nn.Module:
    if module_name == "":
        return root
    cur = root
    for tok in module_name.split("."):
        cur = getattr(cur, tok)
    return cur


def _set_module(root: nn.Module, module_name: str, new_mod: nn.Module) -> None:
    parent_name, child_name = _split_parent_child(module_name)
    parent = _get_module(root, parent_name)
    setattr(parent, child_name, new_mod)


def _head_mask(
    *,
    head_dim: int,
    total_heads: int,
    selected_heads: set[int],
    inverted: bool,
) -> torch.Tensor:
    mask = torch.zeros(total_heads * head_dim, dtype=torch.float32)
    for h in range(total_heads):
        choose = (h in selected_heads)
        if inverted:
            choose = not choose
        if choose:
            s = h * head_dim
            e = s + head_dim
            mask[s:e] = 1.0
    return mask


def _kv_selected(si_query_heads: set[int], n_q: int, n_kv: int) -> set[int]:
    if n_kv <= 0:
        return set()
    if n_kv == n_q:
        return set(si_query_heads)
    q_per_kv = max(1, n_q // max(1, n_kv))
    return {int(h // q_per_kv) for h in si_query_heads if h >= 0}


def _target_mask_for_attention(
    *,
    config,
    module_name: str,
    si_query_heads: set[int],
    selector: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    n_q = int(getattr(config, "num_attention_heads"))
    n_kv = int(getattr(config, "num_key_value_heads", n_q))
    hidden = int(getattr(config, "hidden_size"))
    head_dim = hidden // max(1, n_q)

    if selector == "all":
        return None, None

    invert = selector == "non_si"
    if module_name.endswith("self_attn.q_proj"):
        return None, _head_mask(head_dim=head_dim, total_heads=n_q, selected_heads=si_query_heads, inverted=invert)
    if module_name.endswith("self_attn.k_proj") or module_name.endswith("self_attn.v_proj"):
        kv_sel = _kv_selected(si_query_heads, n_q=n_q, n_kv=n_kv)
        return None, _head_mask(head_dim=head_dim, total_heads=n_kv, selected_heads=kv_sel, inverted=invert)
    if module_name.endswith("self_attn.o_proj"):
        return _head_mask(head_dim=head_dim, total_heads=n_q, selected_heads=si_query_heads, inverted=invert), None
    return None, None


def _supports_head_masking(module_name: str) -> bool:
    """Whether module naming supports explicit per-head in/out masks."""
    return any(
        module_name.endswith(s)
        for s in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    )


def _rank_for_module(
    *,
    condition: str,
    module_name: str,
    has_si_heads: bool,
    hp: LoRAHyperParams,
) -> int:
    if condition == "a_si_protecting_lora":
        if any(module_name.endswith(s) for s in MLP_PROJ_SUFFIXES):
            return int(hp.rank)
        if any(module_name.endswith(s) for s in ATTN_PROJ_SUFFIXES):
            if has_si_heads:
                return int(max(0, hp.si_rank))
            return int(hp.rank)
        return 0

    if condition == "b_uniform_lora":
        return int(hp.rank)

    if condition == "c_si_only_lora":
        if any(module_name.endswith(s) for s in MLP_PROJ_SUFFIXES):
            return 0
        if any(module_name.endswith(s) for s in ATTN_PROJ_SUFFIXES):
            return int(hp.rank if has_si_heads else 0)
        return 0

    if condition == "d_full_qlora_baseline":
        return int(hp.rank)

    if condition == "e_si_amplified_lora":
        if any(module_name.endswith(s) for s in MLP_PROJ_SUFFIXES):
            return int(hp.rank)  # MLP stays at baseline rank
        if any(module_name.endswith(s) for s in ATTN_PROJ_SUFFIXES):
            # Layers with SI heads use CompositeLoRALinear (handled in apply_lora_policy).
            # Layers without SI heads get the reduced non-SI rank for budget matching.
            if has_si_heads:
                return int(hp.si_amplified_rank)  # placeholder; composite handles actual split
            return int(hp.non_si_reduced_rank)
        return 0

    raise ValueError(f"Unknown condition: {condition}")


def _module_targeted_for_condition(*, condition: str, module_name: str, module: nn.Module) -> bool:
    if not _is_linear_like(module):
        return False
    # Condition d is intended as a "full QLoRA-style" calibration baseline.
    # Apply LoRA to every linear projection so it is distinct from condition b
    # (uniform over attention + MLP projections only).
    if condition == "d_full_qlora_baseline":
        return True
    return any(module_name.endswith(s) for s in ALL_SUFFIXES)


def _is_linear_like(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    # HuggingFace GPT-2 uses transformers.pytorch_utils.Conv1D, which behaves
    # like a linear projection with weight shape [in, out].
    cls_name = module.__class__.__name__.lower()
    if "conv1d" in cls_name:
        w = getattr(module, "weight", None)
        return w is not None and int(getattr(w, "ndim", 0)) == 2
    return False


def apply_lora_policy(
    model: nn.Module,
    *,
    condition: str,
    si_heads_by_layer: dict[int, set[int]],
    hp: LoRAHyperParams,
) -> LoRAApplyResult:
    freeze_all_params(model)

    to_replace: list[tuple[str, nn.Module]] = []
    for module_name, module in model.named_modules():
        if not _module_targeted_for_condition(
            condition=condition,
            module_name=module_name,
            module=module,
        ):
            continue
        to_replace.append((module_name, module))

    replaced: list[str] = []
    for module_name, module in to_replace:
        layer_idx = _parse_layer_idx(module_name)
        si_heads = si_heads_by_layer.get(int(layer_idx), set()) if layer_idx is not None else set()
        has_si = len(si_heads) > 0
        is_attn = any(module_name.endswith(s) for s in ATTN_PROJ_SUFFIXES)

        # Condition 'a' with SI heads in attention projections: use composite adapter
        # (full-rank on non-SI heads + rank-1 on SI heads).
        if condition == "a_si_protecting_lora" and is_attn and has_si and _supports_head_masking(module_name):
            in_mask_main, out_mask_main = _target_mask_for_attention(
                config=model.config, module_name=module_name,
                si_query_heads=si_heads, selector="non_si",
            )
            in_mask_si, out_mask_si = _target_mask_for_attention(
                config=model.config, module_name=module_name,
                si_query_heads=si_heads, selector="si_only",
            )
            wrapped = CompositeLoRALinear(
                module,
                rank_main=int(hp.rank),
                rank_si=int(max(1, hp.si_rank)),
                alpha=hp.alpha,
                dropout=hp.dropout,
                in_mask_main=in_mask_main,
                out_mask_main=out_mask_main,
                in_mask_si=in_mask_si,
                out_mask_si=out_mask_si,
            )
            wrapped = wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
            _set_module(model, module_name, wrapped)
            replaced.append(module_name)
            continue

        # Condition 'e' with SI heads in attention projections: use composite adapter
        # (si_amplified_rank on SI heads + non_si_reduced_rank on non-SI heads).
        # This is the inverse of condition 'a': SI heads get MORE capacity.
        if condition == "e_si_amplified_lora" and is_attn and has_si and _supports_head_masking(module_name):
            in_mask_main, out_mask_main = _target_mask_for_attention(
                config=model.config, module_name=module_name,
                si_query_heads=si_heads, selector="non_si",
            )
            in_mask_si, out_mask_si = _target_mask_for_attention(
                config=model.config, module_name=module_name,
                si_query_heads=si_heads, selector="si_only",
            )
            wrapped = CompositeLoRALinear(
                module,
                rank_main=int(max(1, hp.non_si_reduced_rank)),
                rank_si=int(max(1, hp.si_amplified_rank)),
                alpha=hp.alpha,
                dropout=hp.dropout,
                in_mask_main=in_mask_main,
                out_mask_main=out_mask_main,
                in_mask_si=in_mask_si,
                out_mask_si=out_mask_si,
            )
            wrapped = wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
            _set_module(model, module_name, wrapped)
            replaced.append(module_name)
            continue

        rank = _rank_for_module(condition=condition, module_name=module_name, has_si_heads=has_si, hp=hp)
        if rank <= 0:
            continue

        selector = "all"
        if condition == "a_si_protecting_lora" and is_attn:
            # No SI heads in this layer: full-rank on all heads.
            selector = "all"
        elif condition == "e_si_amplified_lora" and is_attn:
            # No SI heads in this layer: reduced-rank on all heads for budget matching.
            selector = "all"
        elif condition == "c_si_only_lora" and is_attn:
            selector = "si_only"

        in_mask = None
        out_mask = None
        if is_attn:
            in_mask, out_mask = _target_mask_for_attention(
                config=model.config,
                module_name=module_name,
                si_query_heads=si_heads,
                selector=selector,
            )

        wrapped = LoRALinear(
            module,
            rank=rank,
            alpha=hp.alpha,
            dropout=hp.dropout,
            in_mask=in_mask,
            out_mask=out_mask,
        )
        wrapped = wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
        _set_module(model, module_name, wrapped)
        replaced.append(module_name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return LoRAApplyResult(replaced_modules=replaced, trainable_params=int(trainable), total_params=int(total))


def collect_trainable_parameter_names(model: nn.Module) -> list[str]:
    out: list[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            out.append(name)
    return out
