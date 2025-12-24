"""
scGPT fine-tuning for reverse perturbation retrieval.

Implements three training modes:
1. Frozen: Inference-only (no training)
2. Head-only: Train retrieval head with frozen backbone
3. LoRA+Head: Train LoRA adapters + retrieval head

Usage:
    python -m src.train.finetune --config src/configs/scgpt_finetune.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence
import pyarrow.parquet as pq
import bisect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import yaml

from .losses import InfoNCELoss, ClassificationLoss
from .lora import (
    apply_lora_to_scgpt,
    get_lora_parameters,
    freeze_model,
    unfreeze_lora,
    count_trainable_parameters,
)


@dataclass
class TrainingConfig:
    """Configuration for scGPT fine-tuning."""
    # New: Tahoe parquet dataset
    parquet_dir: Optional[str] = None   # e.g. /home/user/Desktop/.../tahoe_scgpt_single_target_log1p
    do_binning: bool = False            # 已经 log1p 了，默认关掉 binning
    finetune_checkpoint: Optional[str] = None  # 加载 head/LoRA ckpt
    eval_only: bool = False             # frozen 模式/只评估

    # Data paths
    data_h5ad_path: str = "data/norman/perturb_processed.h5ad"

    # Cell-level split config
    split_min_cells_per_condition: int = 50
    split_query_fraction: float = 0.2
    split_min_query_cells: int = 10
    split_seed: int = 42
    split_output_path: Optional[str] = None

    # Condition-level split config (optional)
    track: Optional[str] = None
    condition_split_train_ratio: float = 0.7
    condition_split_val_ratio: float = 0.1
    condition_split_test_ratio: float = 0.2
    condition_split_seed: int = 42
    condition_split_output_path: Optional[str] = None
    condition_split_n_holdout_genes: int = 5

    # Mode: frozen | head_only | lora_head
    mode: str = "head_only"

    # Loss function: infonce | classification
    loss_fn: str = "infonce"

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    mask_perturbed: bool = True

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"

    # LoRA config
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["out_proj", "linear1", "linear2"]
    )

    # Head config
    head_hidden_dim: int = 256
    head_output_dim: int = 128
    head_dropout: float = 0.2

    # InfoNCE config
    infonce_temperature: float = 0.07

    # Classification config
    label_smoothing: float = 0.1

    # Paths
    checkpoint_dir: str = "model/scgpt_finetune"
    scgpt_model_dir: str = "model/scGPT"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        config_dict = {}
        if "data" in data:
            h5ad_path = data["data"].get("h5ad_path")
            if h5ad_path:
                config_dict["data_h5ad_path"] = h5ad_path
        if "split" in data:
            split_cfg = data["split"]
            config_dict["split_min_cells_per_condition"] = split_cfg.get(
                "min_cells_per_condition",
                config_dict.get("split_min_cells_per_condition"),
            )
            config_dict["split_query_fraction"] = split_cfg.get(
                "query_fraction", config_dict.get("split_query_fraction")
            )
            config_dict["split_min_query_cells"] = split_cfg.get(
                "min_query_cells", config_dict.get("split_min_query_cells")
            )
            config_dict["split_seed"] = split_cfg.get(
                "seed", config_dict.get("split_seed")
            )
            config_dict["split_output_path"] = split_cfg.get("output_path")
        if "track" in data:
            config_dict["track"] = data.get("track")
        if "condition_split" in data:
            cond_cfg = data["condition_split"]
            config_dict["condition_split_train_ratio"] = cond_cfg.get(
                "train_ratio", config_dict.get("condition_split_train_ratio")
            )
            config_dict["condition_split_val_ratio"] = cond_cfg.get(
                "val_ratio", config_dict.get("condition_split_val_ratio")
            )
            config_dict["condition_split_test_ratio"] = cond_cfg.get(
                "test_ratio", config_dict.get("condition_split_test_ratio")
            )
            config_dict["condition_split_seed"] = cond_cfg.get(
                "seed", config_dict.get("condition_split_seed")
            )
            config_dict["condition_split_output_path"] = cond_cfg.get("output_path")
            if "n_holdout_genes" in cond_cfg:
                config_dict["condition_split_n_holdout_genes"] = cond_cfg.get(
                    "n_holdout_genes"
                )
        if "training" in data:
            config_dict.update(data["training"])
        if "lora" in data:
            for k, v in data["lora"].items():
                config_dict[f"lora_{k}"] = v
        if "head" in data:
            for k, v in data["head"].items():
                config_dict[f"head_{k}"] = v
        if "model" in data:
            # Support both pretrained_dir and checkpoint keys
            pretrained = data["model"].get("pretrained_dir") or data["model"].get(
                "checkpoint"
            )
            if pretrained:
                config_dict["scgpt_model_dir"] = pretrained

        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class RetrievalHead(nn.Module):
    """
    MLP projection head for retrieval embeddings.

    Projects scGPT CLS embeddings to a retrieval-optimized space.
    Architecture: [embsize -> hidden -> output_dim]
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
        normalize: bool = True,
    ):
        """
        Initialize retrieval head.

        Args:
            input_dim: Input embedding dimension (scGPT embsize)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
            normalize: Whether to L2-normalize output
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.normalize = normalize

        # Optional learnable temperature for similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.

        Args:
            x: Input embeddings [B, input_dim]

        Returns:
            Projected embeddings [B, output_dim]
        """
        x = self.mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x


class CellEmbeddingDataset(Dataset):
    """
    Dataset for cell embeddings with condition labels.

    Stores pre-computed embeddings for efficient training.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        condition_labels: np.ndarray,
        condition_to_idx: Dict[str, int],
    ):
        """
        Initialize dataset.

        Args:
            embeddings: Cell embeddings [N, D]
            condition_labels: Condition names for each cell [N]
            condition_to_idx: Mapping from condition name to integer
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(
            [condition_to_idx[c] for c in condition_labels],
            dtype=torch.long,
        )
        self.condition_to_idx = condition_to_idx
        self.idx_to_condition = {v: k for k, v in condition_to_idx.items()}

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class ParquetTokenDataset(Dataset):
    """
    Parquet dataset that:
    - remaps Tahoe token_ids -> scGPT vocab ids (including special tokens)
    - ensures exactly one <cls> at position 0 and expr[0]=pad_value
    - optional target masking (label -> target token id)

    ⚠️ NOTE on caching:
    - Caching the *whole* parquet table can explode memory when shards are large,
      especially with multi-worker DataLoader (each worker has its own cache).
    - Default is row-group caching (small, sequential-friendly) instead of whole-table caching.
    """

    def __init__(
        self,
        parquet_paths: Sequence[str],
        cls_token_id: int,
        pad_value: float = 0.0,
        pad_token_id: int = 0,
        eoc_token_id: Optional[int] = None,
        tahoe2scgpt_json: str = "tahoe/tahoe_tokenid_to_scgptid.json",
        # label -> target_gene_symbol -> scGPT_id
        label2target_scgptid: Optional[Dict[int, int]] = None,
        mask_value: float = 0.0,
        # Backward-compat: cache_tables used to cache the *entire* file table. Default now False.
        cache_tables: bool = False,
        # New: cache last row-group instead (much smaller than entire table)
        cache_row_groups: bool = True,
        # Optional: pre-read row group into memory-mapped buffers (may help HDD/network FS)
        use_memory_map: bool = False,
    ):
        self.paths = sorted(parquet_paths)
        self.cls_token_id = int(cls_token_id)
        self.pad_token_id = int(pad_token_id)
        self.eoc_token_id = int(eoc_token_id) if eoc_token_id is not None else None
        self.pad_value = float(pad_value)

        self.label2target_scgptid = label2target_scgptid or {}
        self.mask_value = float(mask_value)

        # load mapping: Tahoe gene token_id -> scGPT vocab id
        with open(tahoe2scgpt_json, "r", encoding="utf-8") as f:
            self.tahoe_gene_map = {int(k): int(v) for k, v in json.load(f).items()}

        # Tahoe special tokens are almost certainly: <pad>=0, <cls>=1, <eoc>=2
        # We must map them to *scGPT* special ids from vocab/config.
        self.tahoe_special_map = {
            0: self.pad_token_id,
            1: self.cls_token_id,
        }
        if self.eoc_token_id is not None:
            self.tahoe_special_map[2] = self.eoc_token_id

        # ----------------------------
        # Build lightweight index:
        # global idx -> (file_k, row_group_j, row_in_group)
        # ----------------------------
        self._pfs: List[pq.ParquetFile] = []
        self._file_rg_offsets: List[List[int]] = []   # per file: prefix rows per row-group, length = n_rg+1
        self._file_num_rows: List[int] = []

        total = 0
        self.prefix = [0]  # global prefix by file
        for p in self.paths:
            pf = pq.ParquetFile(p, memory_map=use_memory_map)
            self._pfs.append(pf)

            nrg = pf.num_row_groups
            rg_offsets = [0]
            s = 0
            # metadata is cheap to read and avoids loading any data buffers
            for j in range(nrg):
                n = pf.metadata.row_group(j).num_rows
                s += n
                rg_offsets.append(s)

            self._file_rg_offsets.append(rg_offsets)
            self._file_num_rows.append(s)

            total += s
            self.prefix.append(total)

        # ----------------------------
        # Cache policy
        # ----------------------------
        self.cache_tables = bool(cache_tables)
        self.cache_row_groups = bool(cache_row_groups) and (not self.cache_tables)

        self._cached_path = None
        self._cached_table = None

        self._cached_rg_key = None  # (path, rg_idx)
        self._cached_rg_table = None

        # columns we need
        self._cols = ["genes", "expressions", "label"]

    def __len__(self):
        return self.prefix[-1]

    def _remap_tokens(self, genes: np.ndarray, exprs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Remap Tahoe token ids -> scGPT ids.
        - Keep only tokens that can be mapped (special or gene_map).
        - Preserve alignment between genes and exprs.
        """
        out_g = []
        out_x = []

        for g, x in zip(genes.tolist(), exprs.tolist()):
            g = int(g)

            if g in self.tahoe_special_map:
                out_g.append(self.tahoe_special_map[g])
                out_x.append(float(x))
                continue

            mapped = self.tahoe_gene_map.get(g)
            if mapped is None:
                # gene not in scGPT vocab -> drop it
                continue

            out_g.append(mapped)
            out_x.append(float(x))

        if len(out_g) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        return np.asarray(out_g, dtype=np.int64), np.asarray(out_x, dtype=np.float32)

    def _ensure_single_cls(self, genes: np.ndarray, exprs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensure exactly one <cls> at pos 0 and expr[0]=pad_value.
        scGPT takes CLS embedding from token 0.
        """
        if genes.size == 0:
            return genes, exprs

        # if first token is not <cls>, insert
        if int(genes[0]) != self.cls_token_id:
            genes = np.insert(genes, 0, self.cls_token_id).astype(np.int64)
            exprs = np.insert(exprs, 0, self.pad_value).astype(np.float32)
        else:
            # already has <cls>, force expr[0]=pad_value to be safe
            exprs = exprs.astype(np.float32, copy=False)
            exprs[0] = self.pad_value

        # remove any extra <cls> that might appear later (very rare but safe)
        if genes.size > 1:
            mask = np.ones_like(genes, dtype=bool)
            extra = np.where((genes == self.cls_token_id) & (np.arange(genes.size) != 0))[0]
            if extra.size > 0:
                mask[extra] = False
                genes = genes[mask]
                exprs = exprs[mask]

        return genes, exprs

    def _mask_target(self, genes: np.ndarray, exprs: np.ndarray, label: int) -> np.ndarray:
        """
        Mask expression for target gene token to prevent leakage (anti-cheat).
        Official code masks perturbed genes in AnnData; we do token-level analogue.
        """
        target_id = self.label2target_scgptid.get(int(label))
        if target_id is None:
            return exprs

        # skip pos 0 (<cls>)
        idx = np.where(genes[1:] == int(target_id))[0]
        if idx.size > 0:
            j = int(idx[0] + 1)
            exprs[j] = self.mask_value
        return exprs

    def _locate(self, idx: int) -> tuple[int, int, int]:
        """
        Map global idx -> (file_k, row_group_j, row_in_group).
        """
        k = bisect.bisect_right(self.prefix, idx) - 1
        in_file = idx - self.prefix[k]

        rg_offsets = self._file_rg_offsets[k]
        rg = bisect.bisect_right(rg_offsets, in_file) - 1
        row_in_rg = in_file - rg_offsets[rg]
        return k, rg, row_in_rg

    def _read_row_group_table(self, file_k: int, rg: int):
        pf = self._pfs[file_k]
        return pf.read_row_group(rg, columns=self._cols)

    def __getitem__(self, idx):
        file_k, rg, row_in_rg = self._locate(idx)
        path = self.paths[file_k]

        # 1) whole-table cache (NOT recommended for large shards)
        if self.cache_tables:
            if path != self._cached_path:
                table = pq.read_table(path, columns=self._cols)
                self._cached_table = table
                self._cached_path = path
            else:
                table = self._cached_table

            genes = table["genes"][idx - self.prefix[file_k]].as_py()
            exprs = table["expressions"][idx - self.prefix[file_k]].as_py()
            label = int(table["label"][idx - self.prefix[file_k]].as_py())

        # 2) row-group cache (recommended default)
        elif self.cache_row_groups:
            key = (path, int(rg))
            if key != self._cached_rg_key:
                rg_table = self._read_row_group_table(file_k, rg)
                self._cached_rg_table = rg_table
                self._cached_rg_key = key
            else:
                rg_table = self._cached_rg_table

            genes = rg_table["genes"][row_in_rg].as_py()
            exprs = rg_table["expressions"][row_in_rg].as_py()
            label = int(rg_table["label"][row_in_rg].as_py())

        # 3) no cache: read exactly one row-group, use one row, then drop
        else:
            rg_table = self._read_row_group_table(file_k, rg)
            genes = rg_table["genes"][row_in_rg].as_py()
            exprs = rg_table["expressions"][row_in_rg].as_py()
            label = int(rg_table["label"][row_in_rg].as_py())

        genes = np.asarray(genes, dtype=np.int64)
        exprs = np.asarray(exprs, dtype=np.float32)

        # 1) remap Tahoe ids -> scGPT ids (incl specials)
        genes, exprs = self._remap_tokens(genes, exprs)
        if genes.size == 0 or genes.size != exprs.size:
            # return a minimal dummy; better is to filter at DataLoader level
            genes = np.asarray([self.cls_token_id], dtype=np.int64)
            exprs = np.asarray([self.pad_value], dtype=np.float32)

        # 2) ensure single CLS at pos0 + expr[0]=pad_value
        genes, exprs = self._ensure_single_cls(genes, exprs)

        # 3) target masking (optional)
        if self.label2target_scgptid:
            exprs = self._mask_target(genes, exprs, label)

        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(exprs).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class CellTokenDataset(Dataset):
    """Dataset that returns tokenized genes/expressions with labels."""

    def __init__(
        self,
        adata,
        gene_ids: np.ndarray,
        condition_labels: np.ndarray,
        condition_to_idx: Dict[str, int],
        cls_token_id: int,
        pad_value: float,
    ):
        self.count_matrix = (
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        )
        self.gene_ids = gene_ids
        self.labels = torch.tensor(
            [condition_to_idx[c] for c in condition_labels],
            dtype=torch.long,
        )
        self.cls_token_id = cls_token_id
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.count_matrix)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx].astype(np.float32)
        genes = self.gene_ids[nonzero_idx].astype(np.int64)
        genes = np.insert(genes, 0, self.cls_token_id)
        values = np.insert(values, 0, self.pad_value).astype(np.float32)
        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
            "label": self.labels[idx],
        }


def _prepare_scgpt_adata(adata, vocab, gene_col: str):
    """Add vocab ids and filter to genes present in scGPT vocab."""
    adata = adata.copy()
    gene_names = (
        adata.var[gene_col].values if gene_col in adata.var else adata.var.index.values
    )
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in gene_names]
    valid_genes = adata.var["id_in_vocab"] >= 0
    n_valid = int(valid_genes.sum())
    n_total = len(valid_genes)
    if n_valid < n_total:
        print(f"  [scGPT] Using {n_valid}/{n_total} genes in vocabulary")
    adata = adata[:, valid_genes].copy()
    gene_ids = np.array(adata.var["id_in_vocab"])
    return adata, gene_ids


def _make_scgpt_collate_fn(collator):
    """Attach labels to scGPT data collator output."""

    def collate(examples):
        labels = torch.stack([ex["label"] for ex in examples])
        base_examples = [
            {"genes": ex["genes"], "expressions": ex["expressions"]} for ex in examples
        ]
        batch = collator(base_examples)
        batch["labels"] = labels
        return batch

    return collate


class FineTunableScGPTEncoder(nn.Module):
    """
    scGPT encoder with fine-tuning support.

    Wraps the scGPT TransformerModel and adds:
    - Retrieval head for embedding projection
    - LoRA adapters for parameter-efficient fine-tuning
    - Multiple training modes (frozen, head_only, lora_head)
    """

    def __init__(
        self,
        scgpt_model: nn.Module,
        config: TrainingConfig,
        num_conditions: Optional[int] = None,
    ):
        """
        Initialize fine-tunable encoder.

        Args:
            scgpt_model: Pretrained scGPT TransformerModel
            config: Training configuration
            num_conditions: Number of conditions (for classification loss)
        """
        super().__init__()

        self.scgpt_model = scgpt_model
        self.config = config

        # Get embedding dimension from model config
        self.embsize = scgpt_model.d_model if hasattr(scgpt_model, "d_model") else 512

        # Retrieval head
        self.retrieval_head = RetrievalHead(
            input_dim=self.embsize,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
            dropout=config.head_dropout,
        )

        # Loss function
        if config.loss_fn == "infonce":
            self.loss_fn = InfoNCELoss(
                temperature=config.infonce_temperature,
                normalize=True,
            )
        elif config.loss_fn == "classification":
            if num_conditions is None:
                raise ValueError("num_conditions required for classification loss")
            self.loss_fn = ClassificationLoss(
                num_conditions=num_conditions,
                embedding_dim=config.head_output_dim,
                hidden_dim=config.head_hidden_dim,
                dropout=config.head_dropout,
                label_smoothing=config.label_smoothing,
            )
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

        # Apply training mode
        self._apply_training_mode()

    def _apply_training_mode(self):
        """Configure model for training mode."""
        mode = self.config.mode

        if mode == "frozen":
            # Freeze everything - inference only
            freeze_model(self.scgpt_model)
            freeze_model(self.retrieval_head)
            freeze_model(self.loss_fn)

        elif mode == "head_only":
            # Freeze backbone, train head + loss
            freeze_model(self.scgpt_model)
            # Head is trainable by default

        elif mode == "lora_head":
            # Freeze backbone, add LoRA, train LoRA + head + loss
            freeze_model(self.scgpt_model)
            apply_lora_to_scgpt(
                self.scgpt_model,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            unfreeze_lora(self.scgpt_model)

        else:
            raise ValueError(f"Unknown training mode: {mode}")

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = count_trainable_parameters(self)
        print(f"[FineTunable] Mode: {mode}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params / total_params:.4%}")

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through retrieval head.

        Note: This assumes embeddings are already extracted from scGPT.
        For end-to-end training with gene tokens, use encode_cells().

        Args:
            embeddings: Cell embeddings from scGPT [B, embsize]

        Returns:
            Projected embeddings [B, output_dim]
        """
        return self.retrieval_head(embeddings)

    def encode_tokens(
        self,
        input_gene_ids: torch.Tensor,
        expressions: torch.Tensor,
        pad_token_id: int,
        batch_labels: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode gene/expression tokens into CLS embeddings."""
        src_key_padding_mask = input_gene_ids.eq(pad_token_id)
        layer_output = self.scgpt_model._encode(
            input_gene_ids,
            expressions,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels,
        )
        embeddings = layer_output[:, 0, :]
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            embeddings: Projected embeddings [B, output_dim]
            labels: Condition labels [B]

        Returns:
            Loss tensor
        """
        return self.loss_fn(embeddings, labels)


class ScGPTTrainer:
    """
    Trainer for scGPT fine-tuning.

    Handles:
    - Training loop with validation
    - Early stopping
    - Checkpoint saving
    - Logging
    """

    def __init__(
        self,
        model: FineTunableScGPTEncoder,
        config: TrainingConfig,
        device: str = "cuda",
        is_master: bool = True,
        end_to_end: bool = False,
        pad_token_id: Optional[int] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: FineTunableScGPTEncoder
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.is_master = is_master
        self.end_to_end = end_to_end
        self.pad_token_id = pad_token_id

        # Optimizer only for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            self.optimizer = None
            if self.is_master:
                print("[Trainer] No trainable parameters. Optimizer is disabled (eval/frozen mode).")
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Learning rate scheduler with warmup
        self.scheduler = None  # Set during training

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        if self.optimizer is None:
            raise RuntimeError("Train called but optimizer is None (no trainable parameters).")

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        for batch in dataloader:
            if self.end_to_end:
                input_gene_ids = batch["gene"].to(self.device)
                expressions = batch["expr"].to(self.device)
                labels = batch["labels"].to(self.device)
                # head_only / frozen：backbone 不训练，用 no_grad 省显存
                if base_model.config.mode in ["head_only", "frozen"]:
                    with torch.no_grad():
                        cls_embeddings = base_model.encode_tokens(
                            input_gene_ids, expressions, pad_token_id=self.pad_token_id
                        )
                else:
                    cls_embeddings = base_model.encode_tokens(
                        input_gene_ids, expressions, pad_token_id=self.pad_token_id
                    )

                projected = base_model.retrieval_head(cls_embeddings)
            else:
                embeddings, labels = batch
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                projected = self.model(embeddings)

            self.optimizer.zero_grad()
            loss = base_model.compute_loss(projected, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        if dist.is_initialized():
            stats = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats.tolist()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        for batch in dataloader:
            if self.end_to_end:
                input_gene_ids = batch["gene"].to(self.device)
                expressions = batch["expr"].to(self.device)
                labels = batch["labels"].to(self.device)
                cls_embeddings = base_model.encode_tokens(
                    input_gene_ids,
                    expressions,
                    pad_token_id=self.pad_token_id,
                )
                projected = base_model.retrieval_head(cls_embeddings)
                loss = base_model.compute_loss(projected, labels)
            else:
                embeddings, labels = batch
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                projected = self.model(embeddings)
                loss = base_model.compute_loss(projected, labels)

            total_loss += loss.item()
            num_batches += 1

        if dist.is_initialized():
            stats = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats.tolist()

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        if self.optimizer is None:
            if self.is_master:
                print("[Trainer] Skip training because optimizer is None.")
            return self.history

        # Setup scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
        )

        if self.is_master:
            print(f"\n{'=' * 60}")
            print(f"Starting training: {self.config.mode} mode")
            print(f"Loss function: {self.config.loss_fn}")
            print(f"Epochs: {self.config.epochs}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"{'=' * 60}\n")


        from tqdm import tqdm

        for epoch in range(self.config.epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # 使用tqdm包装train_loader
            train_loader_with_pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{self.config.epochs}",
                total=len(train_loader),
                leave=True,
                disable=not self.is_master
            )
            # Train (需要在train_epoch内部处理batch)
            train_loss = self.train_epoch(train_loader_with_pbar)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)

            if self.is_master:
                print(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    if self.is_master:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        # Save final
        self._save_checkpoint("final")

        if self.is_master:
            print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

        return self.history

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if not self.is_master:
            return

        checkpoint_path = self.checkpoint_dir / f"{name}_{self.config.mode}.pt"
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        # Save only trainable parts
        state = {
            "config": asdict(self.config),
            "retrieval_head": base_model.retrieval_head.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }

        # Save loss function state if it has parameters
        if hasattr(base_model.loss_fn, "state_dict"):
            state["loss_fn"] = base_model.loss_fn.state_dict()

        # Save LoRA weights if applicable
        if self.config.mode == "lora_head":
            lora_state = {}
            for name, module in base_model.scgpt_model.named_modules():
                from .lora import LoRALinear

                if isinstance(module, LoRALinear):
                    lora_state[name] = {
                        "lora_A": module.lora_A.data,
                        "lora_B": module.lora_B.data,
                    }
            state["lora"] = lora_state

        torch.save(state, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device)

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.retrieval_head.load_state_dict(state["retrieval_head"])

        if "loss_fn" in state and hasattr(base_model.loss_fn, "load_state_dict"):
            base_model.loss_fn.load_state_dict(state["loss_fn"])

        if "lora" in state:
            for name, module in base_model.scgpt_model.named_modules():
                from .lora import LoRALinear

                if isinstance(module, LoRALinear) and name in state["lora"]:
                    module.lora_A.data = state["lora"][name]["lora_A"]
                    module.lora_B.data = state["lora"][name]["lora_B"]

        self.history = state.get("history", {})
        self.best_val_loss = state.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint: {path}")

@torch.no_grad()
def eval_accuracy(model, dataloader, device, pad_token_id):
    model.eval()
    base = model.module if hasattr(model, "module") else model
    correct, total = 0, 0
    for batch in dataloader:
        gene = batch["gene"].to(device)
        expr = batch["expr"].to(device)
        labels = batch["labels"].to(device)
        cls = base.encode_tokens(gene, expr, pad_token_id=pad_token_id, normalize=False)
        z = base.retrieval_head(cls)
        pred = base.loss_fn.predict(z)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="scGPT Fine-tuning")
    parser.add_argument("--parquet_dir", type=str, default=None)
    parser.add_argument("--finetune_checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--do_binning", action="store_true")  # 强制开 binning，默认不开
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/scgpt_finetune.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frozen", "head_only", "lora_head"],
        default=None,
        help="Training mode (overrides config)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["infonce", "classification"],
        default=None,
        help="Loss function (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a quick test without full training",
    )
    return parser.parse_args()


def _setup_ddp() -> Tuple[bool, str, int]:
    """Initialize DDP if launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is not available.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True, f"cuda:{local_rank}", int(os.environ.get("RANK", "0"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return False, device, 0


def main():
    """Main entry point for fine-tuning."""
    args = parse_args()

    # Load config
    if Path(args.config).exists():
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Override from CLI
    if args.mode:
        config.mode = args.mode
    if args.loss:
        config.loss_fn = args.loss

    if args.parquet_dir:
        config.parquet_dir = args.parquet_dir
    if args.finetune_checkpoint:
        config.finetune_checkpoint = args.finetune_checkpoint
    if args.eval_only:
        config.eval_only = True
    if args.do_binning:
        config.do_binning = True

    ddp_enabled, device, rank = _setup_ddp()
    is_master = rank == 0

    if is_master:
        print(f"\n{'=' * 60}")
        print("scGPT Fine-tuning for Reverse Perturbation Retrieval")
        print(f"{'=' * 60}")
        print(f"Mode: {config.mode}")
        print(f"Loss: {config.loss_fn}")
        print(f"{'=' * 60}\n")

    if args.dry_run:
        if is_master:
            print("[Dry run] Configuration loaded successfully.")
            print(f"  LoRA rank: {config.lora_rank}")
            print(f"  Head hidden dim: {config.head_hidden_dim}")
            print(f"  Learning rate: {config.learning_rate}")
        return

    # Import heavy dependencies only when needed
    import sys
    from pathlib import Path as PathLib

    # Add scGPT to path
    scgpt_path = PathLib(__file__).parent.parent.parent / "scGPT"
    if str(scgpt_path) not in sys.path:
        sys.path.insert(0, str(scgpt_path))

    # Load scGPT model
    from src.model import ScGPTEncoder

    if is_master:
        print("Loading scGPT model...")
    encoder = ScGPTEncoder(model_dir=config.scgpt_model_dir)
    encoder._load_model()
    scgpt_model = encoder._model

    # Load dataset
    use_parquet = bool(config.parquet_dir)
    if use_parquet:
        import glob
        from scgpt.data_collator import DataCollator

        parquet_dir = config.parquet_dir
        if is_master:
            print(f"Using Tahoe parquet dataset: {parquet_dir}")

        # 1) 读 label vocab -> num_conditions
        vocab_path = os.path.join(parquet_dir, "label_vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_obj = json.load(f)

        # 兼容你之前保存的 gene2y/gene2label 两种 key
        gene2label = vocab_obj.get("gene2label") or vocab_obj.get("gene2y")
        if gene2label is None:
            raise ValueError(f"label_vocab.json missing gene2label/gene2y: {vocab_path}")
        num_conditions = len(gene2label)

        # 2) 取 split parquet
        train_paths = sorted(glob.glob(os.path.join(parquet_dir, "train_*.parquet")))
        val_paths   = sorted(glob.glob(os.path.join(parquet_dir, "val_*.parquet")))
        test_paths  = sorted(glob.glob(os.path.join(parquet_dir, "test_*.parquet")))
        ood_paths   = sorted(glob.glob(os.path.join(parquet_dir, "ood_test_*.parquet")))

        if len(train_paths) == 0:
            raise FileNotFoundError(f"No train_*.parquet under {parquet_dir}")
        if len(val_paths) == 0:
            raise FileNotFoundError(f"No val_*.parquet under {parquet_dir}")
        if len(test_paths) == 0:
            raise FileNotFoundError(f"No test_*.parquet under {parquet_dir}")
        if len(ood_paths) == 0 and is_master:
            print("[Warn] No ood_test_*.parquet found; OOD eval will be skipped.")

        # 3) 建 FineTunableScGPTEncoder（分类头类别数就是 num_conditions）
        model = FineTunableScGPTEncoder(
            scgpt_model=scgpt_model,
            config=config,
            num_conditions=num_conditions,
        )
        #  label -> gene symbol
        with open("./tahoe/tahoe_scgpt_single_target_log1p/label_vocab.json", "r", encoding="utf-8") as f:
            lv = json.load(f)
            label2gene = {int(k): v for k, v in lv["label2gene"].items()}

        # gene symbol -> scGPT id (来自你本地 vocab.json)
        with open("./model/scGPT/vocab.json", "r") as f:
            vocab = json.load(f)
        token2id = vocab["token2id"] if "token2id" in vocab else vocab

        label2target_scgptid = {}
        for lab, g in label2gene.items():
            if g in token2id:
                label2target_scgptid[int(lab)] = int(token2id[g])

        # 4) Token dataset（你的 parquet 已经是 genes/expressions/label）
        train_subset = ParquetTokenDataset(
            train_paths,
            cls_token_id=encoder._vocab["<cls>"],
            pad_value=encoder._model_configs["pad_value"],
            pad_token_id = encoder._vocab[encoder._model_configs["pad_token"]],
            #eoc_token_id=encoder._vocab["<eoc>"],  # 若你的 vocab 里有
            tahoe2scgpt_json="tahoe/tahoe_tokenid_to_scgptid.json",
            label2target_scgptid=label2target_scgptid,
            mask_value=0.0,
        )
        val_subset = ParquetTokenDataset(
            val_paths,
            cls_token_id=encoder._vocab["<cls>"],
            pad_value=encoder._model_configs["pad_value"],
            pad_token_id = encoder._vocab[encoder._model_configs["pad_token"]],
            tahoe2scgpt_json="tahoe/tahoe_tokenid_to_scgptid.json",
            label2target_scgptid=label2target_scgptid,
            mask_value=0.0,
        )
        test_dataset = ParquetTokenDataset(
            test_paths,
            cls_token_id=encoder._vocab["<cls>"],
            pad_value=encoder._model_configs["pad_value"],
            pad_token_id = encoder._vocab[encoder._model_configs["pad_token"]],
            tahoe2scgpt_json="tahoe/tahoe_tokenid_to_scgptid.json",
            label2target_scgptid=label2target_scgptid,
            mask_value=0.0,
        )
        ood_dataset = None
        if len(ood_paths) > 0:
            ood_dataset = ParquetTokenDataset(
                ood_paths,
                cls_token_id=encoder._vocab["<cls>"],
                pad_value=encoder._model_configs["pad_value"],
                pad_token_id = encoder._vocab[encoder._model_configs["pad_token"]],
                tahoe2scgpt_json="tahoe/tahoe_tokenid_to_scgptid.json",
                label2target_scgptid=label2target_scgptid,
                mask_value=0.0,
            )

        # 5) DataCollator：你已经 log1p，建议 do_binning=False（默认）
        pad_token_id = encoder._vocab[encoder._model_configs["pad_token"]]
        collator = DataCollator(
            do_padding=True,
            pad_token_id=pad_token_id,
            pad_value=encoder._model_configs["pad_value"],
            do_mlm=False,
            do_binning=config.do_binning,   # <<<<<< 默认 False
            max_length=encoder.max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        collate_fn = _make_scgpt_collate_fn(collator)

    else:
        # ====== 原来的 h5ad 流程保持不变 ======
        from src.data import load_perturb_data, mask_perturbed_genes
        from src.data import ConditionSplitter, ConditionSplit
        if is_master:
            print("Loading dataset...")
        dataset = load_perturb_data(
            h5ad_path=config.data_h5ad_path,
            split_path=config.split_output_path,
            min_cells_per_condition=config.split_min_cells_per_condition,
            query_fraction=config.split_query_fraction,
            min_query_cells=config.split_min_query_cells,
            seed=config.split_seed,
        )
        if (
            config.split_output_path
            and dataset.split is not None
            and not Path(config.split_output_path).exists()
        ):
            dataset.split.save(config.split_output_path)

        if config.track and config.track != "in_dist":
            cond_split = None
            if (
                config.condition_split_output_path
                and Path(config.condition_split_output_path).exists()
            ):
                cond_split = ConditionSplit.load(config.condition_split_output_path)
            else:
                splitter = ConditionSplitter(
                    train_ratio=config.condition_split_train_ratio,
                    val_ratio=config.condition_split_val_ratio,
                    test_ratio=config.condition_split_test_ratio,
                    seed=config.condition_split_seed,
                )
                if config.track == "unseen_gene":
                    cond_split = splitter.split_unseen_gene(
                        dataset.all_conditions,
                        n_holdout_genes=config.condition_split_n_holdout_genes,
                    )
                else:
                    cond_split = splitter.split(dataset.all_conditions, track=config.track)
                if config.condition_split_output_path:
                    cond_split.save(config.condition_split_output_path)
            dataset.apply_condition_split(cond_split)
        elif config.track == "in_dist" and is_master:
            print("  [Info] in_dist track uses cell-level split; skipping condition split.")

        # Get condition mapping
        conditions = dataset.all_conditions
        condition_to_idx = {c: i for i, c in enumerate(conditions)}
        num_conditions = len(conditions)
        if is_master:
            print(f"  Conditions: {num_conditions}")

        # Create model
        model = FineTunableScGPTEncoder(
            scgpt_model=scgpt_model,
            config=config,
            num_conditions=num_conditions,
        )

        condition_sets = dataset.get_condition_sets()
        train_conditions = condition_sets.get("train") or conditions
        val_conditions = condition_sets.get("val") or []

        if config.mode == "lora_head":
            if is_master:
                print("Preparing token datasets for LoRA fine-tuning...")
            train_adata = dataset.get_ref_adata_for_conditions(train_conditions)
            if config.mask_perturbed:
                train_adata = mask_perturbed_genes(
                    train_adata, condition_col=dataset.condition_col
                )
            train_adata, train_gene_ids = _prepare_scgpt_adata(
                train_adata, encoder._vocab, encoder.gene_col
            )
            train_labels = train_adata.obs[dataset.condition_col].values
            train_dataset = CellTokenDataset(
                adata=train_adata,
                gene_ids=train_gene_ids,
                condition_labels=train_labels,
                condition_to_idx=condition_to_idx,
                cls_token_id=encoder._vocab["<cls>"],
                pad_value=encoder._model_configs["pad_value"],
            )

            if val_conditions:
                val_adata = dataset.get_ref_adata_for_conditions(val_conditions)
                if config.mask_perturbed:
                    val_adata = mask_perturbed_genes(
                        val_adata, condition_col=dataset.condition_col
                    )
                val_adata, val_gene_ids = _prepare_scgpt_adata(
                    val_adata, encoder._vocab, encoder.gene_col
                )
                val_labels = val_adata.obs[dataset.condition_col].values
                val_dataset = CellTokenDataset(
                    adata=val_adata,
                    gene_ids=val_gene_ids,
                    condition_labels=val_labels,
                    condition_to_idx=condition_to_idx,
                    cls_token_id=encoder._vocab["<cls>"],
                    pad_value=encoder._model_configs["pad_value"],
                )
                train_subset = train_dataset
                val_subset = val_dataset
            else:
                n_train = int(0.8 * len(train_dataset))
                n_val = len(train_dataset) - n_train
                generator = torch.Generator().manual_seed(config.split_seed)
                train_subset, val_subset = torch.utils.data.random_split(
                    train_dataset, [n_train, n_val], generator=generator
                )

            from scgpt.data_collator import DataCollator

            pad_token_id = encoder._vocab[encoder._model_configs["pad_token"]]
            collator = DataCollator(
                do_padding=True,
                pad_token_id=pad_token_id,
                pad_value=encoder._model_configs["pad_value"],
                do_mlm=False,
                do_binning=True,
                max_length=encoder.max_length,
                sampling=True,
                keep_first_n_tokens=1,
            )
            collate_fn = _make_scgpt_collate_fn(collator)
        else:
            # Extract embeddings (pre-compute for efficiency)
            if is_master:
                print("Extracting embeddings...")
            train_adata = dataset.get_ref_adata_for_conditions(train_conditions)
            if config.mask_perturbed:
                train_adata = mask_perturbed_genes(
                    train_adata, condition_col=dataset.condition_col
                )
            train_embeddings = encoder.encode_adata(train_adata)
            train_labels = train_adata.obs[dataset.condition_col].values

            train_dataset = CellEmbeddingDataset(
                embeddings=train_embeddings,
                condition_labels=train_labels,
                condition_to_idx=condition_to_idx,
            )

            if val_conditions:
                val_adata = dataset.get_ref_adata_for_conditions(val_conditions)
                if config.mask_perturbed:
                    val_adata = mask_perturbed_genes(
                        val_adata, condition_col=dataset.condition_col
                    )
                val_embeddings = encoder.encode_adata(val_adata)
                val_labels = val_adata.obs[dataset.condition_col].values
                val_dataset = CellEmbeddingDataset(
                    embeddings=val_embeddings,
                    condition_labels=val_labels,
                    condition_to_idx=condition_to_idx,
                )
                train_subset = train_dataset
                val_subset = val_dataset
            else:
                n_train = int(0.8 * len(train_dataset))
                n_val = len(train_dataset) - n_train
                generator = torch.Generator().manual_seed(config.split_seed)
                train_subset, val_subset = torch.utils.data.random_split(
                    train_dataset, [n_train, n_val], generator=generator
                )
            collate_fn = None
            pad_token_id = None

    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_sampler = torch.utils.data.DistributedSampler(train_subset, shuffle=False)
        val_sampler = torch.utils.data.DistributedSampler(val_subset, shuffle=False)

    loader_kwargs = dict(
        num_workers=4,              # 先 4；CPU 多可试 8
        pin_memory=True,
        persistent_workers=True,    # PyTorch>=1.7
        prefetch_factor=2,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    # Train
    if ddp_enabled:
        allow_unused = config.loss_fn == "infonce" or config.mode == "lora_head"
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[int(device.split(":")[-1])],
            find_unused_parameters=allow_unused,
        )
    end_to_end_flag = use_parquet and (config.mode in ["head_only", "lora_head", "frozen"])
    trainer = ScGPTTrainer(
        model,
        config,
        device=device,
        is_master=is_master,
        end_to_end=end_to_end_flag,
        pad_token_id=pad_token_id if end_to_end_flag else None,
    )

    if use_parquet and (config.mode == "frozen" or config.eval_only):
        if not config.finetune_checkpoint:
            raise ValueError("frozen/eval_only requires --finetune_checkpoint to load head/LoRA weights")

        trainer.load_checkpoint(config.finetune_checkpoint)

        # test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        acc_test = eval_accuracy(trainer.model, test_loader, device, pad_token_id)

        acc_ood = None
        if ood_dataset is not None:
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            acc_ood = eval_accuracy(trainer.model, ood_loader, device, pad_token_id)

        if is_master:
            print(f"[Eval] test acc: {acc_test:.4f}")
            if acc_ood is not None:
                print(f"[Eval] ood  acc: {acc_ood:.4f}")
        return

    # Save history
    history = trainer.train(train_loader, val_loader)
    if is_master:
        history_path = Path(config.checkpoint_dir) / f"history_{config.mode}.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved: {history_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
