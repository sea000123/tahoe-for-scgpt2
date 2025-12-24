"""
scGPT fine-tuning for reverse perturbation retrieval (Tahoe-only).

Supported training modes:
1) frozen     : inference-only (no training)
2) head_only  : train retrieval head (and loss head) with frozen backbone
3) lora_head  : train LoRA adapters + retrieval head (and loss head)

This script is **Tahoe parquet only** and removes the original Norman (h5ad) pipeline.

Example:
    python -m src.train.finetune \
      --mode head_only \
      --loss classification \
      --parquet_dir /tahoe/tahoe_scgpt_single_target_log1p
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import time
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime
from collections import defaultdict

# reuse uploaded eval utilities
from src.evaluate.metrics import compute_all_metrics  # :contentReference[oaicite:8]{index=8}
from src.evaluate.confidence import ConfidenceScorer, coverage_accuracy_curve, compute_auc_coverage_accuracy  # :contentReference[oaicite:9]{index=9}
from src.evaluate.error_analysis import generate_error_report  # :contentReference[oaicite:10]{index=10}
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from .losses import InfoNCELoss, ClassificationLoss
from .lora import (
    apply_lora_to_scgpt,
    freeze_model,
    unfreeze_lora,
    count_trainable_parameters,
)


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainingConfig:
    """Configuration for scGPT fine-tuning (Tahoe parquet only)."""

    # Data
    parquet_dir: Optional[str] = None  # e.g. /path/to/tahoe_scgpt_single_target_log1p
    do_binning: bool = False           # Tahoe build script already does log1p; default off
    finetune_checkpoint: Optional[str] = None  # load head/LoRA ckpt for eval/frozen
    eval_only: bool = False            # only evaluate (requires checkpoint)

    # Mode: frozen | head_only | lora_head
    mode: str = "frozen"

    # Loss function: infonce | classification
    loss_fn: str = "infonce"

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

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

    # Tahoe mapping file (Tahoe token_id -> scGPT vocab id)
    tahoe2scgpt_json: str = "tahoe/tahoe_tokenid_to_scgptid.json"


# ----------------------------
# Model components
# ----------------------------
class RetrievalHead(nn.Module):
    """MLP projection head for retrieval embeddings."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
        normalize: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.normalize = normalize
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

class EmbeddingDataset(Dataset):
    def __init__(self, z_path: Path, y_path: Path):
        self.z = np.load(z_path)  # [N, D]
        self.y = np.load(y_path)  # [N]
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, idx):
        return {
            "z": torch.from_numpy(self.z[idx]).float(),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.long),
        }

def emb_collate(batch):
    z = torch.stack([b["z"] for b in batch], dim=0)
    y = torch.stack([b["label"] for b in batch], dim=0)
    return {"z": z, "labels": y}


class ParquetTokenDataset(Dataset):
    """
    Tahoe parquet dataset -> scGPT token tensors.

    Expects parquet rows to contain:
      - genes: List[int]          (Tahoe token IDs; may or may not include special tokens)
      - expressions: List[float]  (already log1p in your build script)
      - label: int                (target class)

    It remaps Tahoe token_ids -> scGPT vocab IDs (including special tokens),
    ensures exactly one <cls> at position 0, and optionally masks the
    target gene token expression to reduce leakage.
    """

    def __init__(
        self,
        parquet_paths: Sequence[str],
        cls_token_id: int,
        pad_value: float = 0.0,
        pad_token_id: int = 0,
        eoc_token_id: Optional[int] = None,
        tahoe2scgpt_json: str = "tahoe/tahoe_tokenid_to_scgptid.json",
        remap_tahoe_to_scgpt: bool = True,
        label2target_scgptid: Optional[Dict[int, int]] = None,
        mask_value: float = 0.0,
        cache_tables: bool = False,
        cache_row_groups: bool = True,
        use_memory_map: bool = False,
    ):
        self.paths = sorted(parquet_paths)
        self.cls_token_id = int(cls_token_id)
        self.pad_token_id = int(pad_token_id)
        self.eoc_token_id = int(eoc_token_id) if eoc_token_id is not None else None
        self.pad_value = float(pad_value)

        self.label2target_scgptid = label2target_scgptid or {}
        self.mask_value = float(mask_value)

        self.remap_tahoe_to_scgpt = bool(remap_tahoe_to_scgpt)
        if self.remap_tahoe_to_scgpt:
            with open(tahoe2scgpt_json, "r", encoding="utf-8") as f:
                self.tahoe_gene_map = {int(k): int(v) for k, v in json.load(f).items()}
        else:
            # genes in parquet are already scGPT vocab ids
            self.tahoe_gene_map = {}
        # Tahoe special tokens are typically: <pad>=0, <cls>=1, <eoc>=2
        # Only needed when we are remapping Tahoe ids -> scGPT ids at training time.
        if self.remap_tahoe_to_scgpt:
            self.tahoe_special_map = {
                0: self.pad_token_id,
                1: self.cls_token_id,
            }
            if self.eoc_token_id is not None:
                self.tahoe_special_map[2] = self.eoc_token_id
        else:
            self.tahoe_special_map = {}

        # Lightweight index: global idx -> (file_k, row_group_j, row_in_group)
        self._pfs: List[pq.ParquetFile] = []
        self._file_rg_offsets: List[List[int]] = []
        self._file_num_rows: List[int] = []

        total = 0
        self.prefix = [0]
        for p in self.paths:
            pf = pq.ParquetFile(p, memory_map=use_memory_map)
            self._pfs.append(pf)

            nrg = pf.num_row_groups
            rg_offsets = [0]
            s = 0
            for j in range(nrg):
                n = pf.metadata.row_group(j).num_rows
                s += n
                rg_offsets.append(s)

            self._file_rg_offsets.append(rg_offsets)
            self._file_num_rows.append(s)

            total += s
            self.prefix.append(total)

        self.cache_tables = bool(cache_tables)
        self.cache_row_groups = bool(cache_row_groups) and (not self.cache_tables)

        self._cached_path = None
        self._cached_table = None

        self._cached_rg_key = None  # (path, rg_idx)
        self._cached_rg_table = None

        self._cols = ["genes", "expressions", "label"]

    def __len__(self):
        return self.prefix[-1]

    def _remap_tokens(self, genes: np.ndarray, exprs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.remap_tahoe_to_scgpt:
            return genes.astype(np.int64, copy=False), exprs.astype(np.float32, copy=False)
        out_g, out_x = [], []
        for g, x in zip(genes.tolist(), exprs.tolist()):
            g = int(g)

            if g in self.tahoe_special_map:
                out_g.append(self.tahoe_special_map[g])
                out_x.append(float(x))
                continue

            mapped = self.tahoe_gene_map.get(g)
            if mapped is None:
                continue
            out_g.append(mapped)
            out_x.append(float(x))

        if len(out_g) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        return np.asarray(out_g, dtype=np.int64), np.asarray(out_x, dtype=np.float32)

    def _ensure_single_cls(self, genes: np.ndarray, exprs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if genes.size == 0:
            return genes, exprs

        if int(genes[0]) != self.cls_token_id:
            genes = np.insert(genes, 0, self.cls_token_id).astype(np.int64)
            exprs = np.insert(exprs, 0, self.pad_value).astype(np.float32)
        else:
            exprs = exprs.astype(np.float32, copy=False)
            exprs[0] = self.pad_value

        if genes.size > 1:
            extra = np.where((genes == self.cls_token_id) & (np.arange(genes.size) != 0))[0]
            if extra.size > 0:
                mask = np.ones_like(genes, dtype=bool)
                mask[extra] = False
                genes = genes[mask]
                exprs = exprs[mask]

        return genes, exprs

    def _mask_target(self, genes: np.ndarray, exprs: np.ndarray, label: int) -> np.ndarray:
        target_id = self.label2target_scgptid.get(int(label))
        if target_id is None:
            return exprs
        idx = np.where(genes[1:] == int(target_id))[0]  # skip CLS
        if idx.size > 0:
            j = int(idx[0] + 1)
            exprs[j] = self.mask_value
        return exprs

    def _locate(self, idx: int) -> tuple[int, int, int]:
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

        if self.cache_tables:
            if path != self._cached_path:
                table = pq.read_table(path, columns=self._cols)
                self._cached_table = table
                self._cached_path = path
            else:
                table = self._cached_table

            row_idx = idx - self.prefix[file_k]
            genes = table["genes"][row_idx].as_py()
            exprs = table["expressions"][row_idx].as_py()
            label = int(table["label"][row_idx].as_py())

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

        else:
            rg_table = self._read_row_group_table(file_k, rg)
            genes = rg_table["genes"][row_in_rg].as_py()
            exprs = rg_table["expressions"][row_in_rg].as_py()
            label = int(rg_table["label"][row_in_rg].as_py())

        genes = np.asarray(genes, dtype=np.int64)
        exprs = np.asarray(exprs, dtype=np.float32)

        genes, exprs = self._remap_tokens(genes, exprs)
        if genes.size == 0 or genes.size != exprs.size:
            genes = np.asarray([self.cls_token_id], dtype=np.int64)
            exprs = np.asarray([self.pad_value], dtype=np.float32)

        genes, exprs = self._ensure_single_cls(genes, exprs)

        if self.label2target_scgptid:
            exprs = self._mask_target(genes, exprs, label)

        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(exprs).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def _make_scgpt_collate_fn(collator):
    """Attach labels to scGPT DataCollator output."""

    def collate(examples):
        labels = torch.stack([ex["label"] for ex in examples])
        base_examples = [{"genes": ex["genes"], "expressions": ex["expressions"]} for ex in examples]
        batch = collator(base_examples)
        batch["labels"] = labels
        return batch

    return collate

@torch.no_grad()
def build_embedding_cache(
    *,
    model,
    device: str,
    pad_token_id: int,
    dataset: torch.utils.data.Dataset,
    collate_fn,
    split_name: str,
    out_dir: Path,
    batch_size: int,
):
    """
    Save:
    - z: projected embedding after retrieval_head  [N, D]
    - y: labels                                  [N]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    z_path = out_dir / f"{split_name}_z.npy"
    y_path = out_dir / f"{split_name}_y.npy"

    if z_path.exists() and y_path.exists():
        print(f"[Cache] Found existing cache: {z_path} / {y_path}")
        return z_path, y_path

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    base = model.module if hasattr(model, "module") else model
    base.eval()

    zs, ys = [], []
    for i, batch in enumerate(tqdm(loader, desc=f"[Cache:{split_name}]", total=len(loader))):
        gene = batch["gene"].to(device, non_blocking=True)
        expr = batch["expr"].to(device, non_blocking=True)
        labels = batch["labels"].cpu().long()

        cls = base.encode_tokens(gene, expr, pad_token_id=pad_token_id, normalize=False)
        z = base.retrieval_head(cls)
        z = F.normalize(z, p=2, dim=-1).detach().cpu().float()

        zs.append(z)
        ys.append(labels)
        if i % 200 == 0:
            print(f"[Cache:{split_name}] step={i}/{len(loader)}")

    z_all = torch.cat(zs, dim=0).numpy()
    y_all = torch.cat(ys, dim=0).numpy()

    np.save(z_path, z_all)
    np.save(y_path, y_all)
    print(f"[Cache] Saved: {z_path} shape={z_all.shape}, {y_path} shape={y_all.shape}")
    return z_path, y_path

class FineTunableScGPTEncoder(nn.Module):
    """scGPT encoder wrapper + retrieval head + (optional) LoRA adapters."""

    def __init__(
        self,
        scgpt_model: nn.Module,
        config: TrainingConfig,
        num_conditions: Optional[int] = None,
    ):
        super().__init__()
        self.scgpt_model = scgpt_model
        self.config = config

        self.embsize = scgpt_model.d_model if hasattr(scgpt_model, "d_model") else 512

        self.retrieval_head = RetrievalHead(
            input_dim=self.embsize,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
            dropout=config.head_dropout,
        )

        if config.loss_fn == "infonce":
            self.loss_fn = InfoNCELoss(temperature=config.infonce_temperature, normalize=True)
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

        self._apply_training_mode()

    def _apply_training_mode(self):
        mode = self.config.mode

        if mode == "frozen":
            freeze_model(self.scgpt_model)
            freeze_model(self.retrieval_head)
            freeze_model(self.loss_fn)

        elif mode == "head_only":
            freeze_model(self.scgpt_model)

        elif mode == "lora_head":
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

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = count_trainable_parameters(self)
        print(f"[FineTunable] Mode: {mode}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params / total_params:.4%}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.retrieval_head(embeddings)

    def encode_tokens(
        self,
        input_gene_ids: torch.Tensor,
        expressions: torch.Tensor,
        pad_token_id: int,
        batch_labels: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
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

    def compute_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(embeddings, labels)


# ----------------------------
# Trainer
# ----------------------------
class ScGPTTrainer:
    def __init__(
        self,
        model: FineTunableScGPTEncoder,
        config: TrainingConfig,
        device: str = "cuda",
        is_master: bool = True,
        end_to_end: bool = True,
        pad_token_id: Optional[int] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.is_master = is_master
        self.end_to_end = end_to_end
        self.pad_token_id = pad_token_id

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            self.optimizer = None
            if self.is_master:
                print("[Trainer] No trainable parameters. Optimizer disabled (eval/frozen mode).")
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        self.scheduler = None
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        if self.optimizer is None:
            raise RuntimeError("Train called but optimizer is None.")

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        # ---- timing accumulators ----
        t_data = 0.0
        t_fwd = 0.0
        t_bwd = 0.0

        # for dataloader timing: time between iterations
        prev_end = time.perf_counter()

        for step, batch in enumerate(dataloader):
            if getattr(self.config, "max_steps", 0) and step + 1 >= self.config.max_steps:
                break

            # ---- data time: time waiting for the next batch ----
            now = time.perf_counter()
            t_data += (now - prev_end)
            if "z" in batch:
                z = batch["z"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                t0 = time.perf_counter()
                projected = z  # already projected
                loss = base_model.compute_loss(projected, labels)
            else:
                # 原来的 token 路径（保持不变）
                # Move to device
                input_gene_ids = batch["gene"].to(self.device, non_blocking=True)
                expressions = batch["expr"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # ---- forward time ----
                torch.cuda.synchronize() if "cuda" in self.device else None
                t0 = time.perf_counter()

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
                loss = base_model.compute_loss(projected, labels)

            torch.cuda.synchronize() if "cuda" in self.device else None
            t1 = time.perf_counter()
            t_fwd += (t1 - t0)

            # ---- backward/step time ----
            torch.cuda.synchronize() if "cuda" in self.device else None
            t2 = time.perf_counter()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            torch.cuda.synchronize() if "cuda" in self.device else None
            t3 = time.perf_counter()
            t_bwd += (t3 - t2)

            total_loss += float(loss.item())
            num_batches += 1

            # update prev_end for next iteration timing
            prev_end = time.perf_counter()

            # ---- optional periodic logging ----
            if self.is_master and step in (0, 10, 50, 100):
                if "z" in batch:
                    print(f"[Time] step={step} data=... fwd=... bwd=... (cached_z)")
                else:
                            # effective tokens & pad ratio for this batch
                    tok = input_gene_ids.numel()
                    nonpad = input_gene_ids.ne(self.pad_token_id).sum().item()
                    pad_ratio = 1 - (nonpad / max(tok, 1))
                    print(
                        f"[Time] step={step} "
                        f"data={t_data/num_batches:.4f}s "
                        f"fwd={t_fwd/num_batches:.4f}s "
                        f"bwd={t_bwd/num_batches:.4f}s "
                        f"pad_ratio={pad_ratio:.2%}"
                    )

        if dist.is_initialized():
            stats = torch.tensor([total_loss, num_batches, t_data, t_fwd, t_bwd], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches, t_data, t_fwd, t_bwd = stats.tolist()

        # ---- epoch summary ----
        if self.is_master:
            denom = max(num_batches, 1)
            print(
                f"[EpochTiming] avg_per_step: data={t_data/denom:.4f}s "
                f"fwd={t_fwd/denom:.4f}s bwd={t_bwd/denom:.4f}s "
                f"(total={ (t_data+t_fwd+t_bwd)/denom:.4f}s)"
            )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        for batch in dataloader:
            input_gene_ids = batch["gene"].to(self.device)
            expressions = batch["expr"].to(self.device)
            labels = batch["labels"].to(self.device)

            cls_embeddings = base_model.encode_tokens(
                input_gene_ids, expressions, pad_token_id=self.pad_token_id
            )
            projected = base_model.retrieval_head(cls_embeddings)
            loss = base_model.compute_loss(projected, labels)

            total_loss += loss.item()
            num_batches += 1

        if dist.is_initialized():
            stats = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats.tolist()

        return total_loss / max(num_batches, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        if self.optimizer is None:
            if self.is_master:
                print("[Trainer] Skip training because optimizer is None.")
            return self.history
        print("len(train_loader):", len(train_loader))
        print("self.config.batch_size:", self.config.batch_size)
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
        )

        if self.is_master:
            print("\n" + "=" * 60)
            print(f"Starting training: {self.config.mode} mode")
            print(f"Loss function: {self.config.loss_fn}")
            print(f"Epochs: {self.config.epochs}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Learning rate: {self.config.learning_rate}")
            print("=" * 60 + "\n")

        for epoch in range(self.config.epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loader_with_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config.epochs}",
                total=len(train_loader),
                leave=True,
                disable=not self.is_master,
            )
            train_loss = self.train_epoch(train_loader_with_pbar)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.is_master:
                print(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

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

        self._save_checkpoint("final")
        if self.is_master:
            print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def _save_checkpoint(self, name: str):
        if not self.is_master:
            return

        checkpoint_path = self.checkpoint_dir / f"{name}_{self.config.mode}.pt"
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        state = {
            "config": asdict(self.config),
            "retrieval_head": base_model.retrieval_head.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        if hasattr(base_model.loss_fn, "state_dict"):
            state["loss_fn"] = base_model.loss_fn.state_dict()

        if self.config.mode == "lora_head":
            lora_state = {}
            for mod_name, module in base_model.scgpt_model.named_modules():
                from .lora import LoRALinear
                if isinstance(module, LoRALinear):
                    lora_state[mod_name] = {
                        "lora_A": module.lora_A.data,
                        "lora_B": module.lora_B.data,
                    }
            state["lora"] = lora_state

        torch.save(state, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        base_model.retrieval_head.load_state_dict(state["retrieval_head"])

        if "loss_fn" in state and hasattr(base_model.loss_fn, "load_state_dict"):
            base_model.loss_fn.load_state_dict(state["loss_fn"])

        if "lora" in state:
            for mod_name, module in base_model.scgpt_model.named_modules():
                from .lora import LoRALinear
                if isinstance(module, LoRALinear) and mod_name in state["lora"]:
                    module.lora_A.data = state["lora"][mod_name]["lora_A"]
                    module.lora_B.data = state["lora"][mod_name]["lora_B"]

        self.history = state.get("history", {})
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint: {path}")


@torch.no_grad()
def _encode_projected_embeddings(model, dataloader, device, pad_token_id):
    """
    Encode a dataloader into projected retrieval embeddings (z) + int labels.

    Returns:
        z: (N, D) float32 numpy
        y: (N,) int64 numpy
    """
    model.eval()
    base = model.module if hasattr(model, "module") else model

    zs = []
    ys = []
    for batch in dataloader:
        gene = batch["gene"].to(device)
        expr = batch["expr"].to(device)
        labels = batch["labels"].to(device)

        cls = base.encode_tokens(gene, expr, pad_token_id=pad_token_id, normalize=False)
        z = base.retrieval_head(cls)  # (B, D)
        # ensure normalized for cosine
        z = F.normalize(z, p=2, dim=-1)

        zs.append(z.detach().cpu())
        ys.append(labels.detach().cpu())

    z_all = torch.cat(zs, dim=0).float().numpy() if zs else np.zeros((0, 1), dtype=np.float32)
    y_all = torch.cat(ys, dim=0).long().numpy() if ys else np.zeros((0,), dtype=np.int64)
    return z_all, y_all


def _build_prototype_library(
    z_train: np.ndarray,
    y_train: np.ndarray,
    label2cond: Dict[int, str],
    library_type: str = "mean",
    n_prototypes: int = 30,
    m_samples: int = 50,
    seed: int = 42,
):
    """
    Build class prototypes in embedding space.

    Returns:
        proto_vecs: (M, D)
        proto_labels: list[str] length M (condition string per prototype)
        candidate_conditions: sorted unique condition strings
    """
    if z_train.size == 0:
        return np.zeros((0, 1), dtype=np.float32), [], []

    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for z, y in zip(z_train, y_train):
        by_label[int(y)].append(z)

    proto_vecs = []
    proto_labels = []

    for lab, vecs in by_label.items():
        cond = label2cond.get(int(lab), str(lab))
        X = np.stack(vecs, axis=0)  # (n, d)

        if library_type == "mean":
            p = X.mean(axis=0)
            proto_vecs.append(p)
            proto_labels.append(cond)
        elif library_type == "bootstrap":
            n = X.shape[0]
            sample_size = min(m_samples, n) if m_samples and m_samples > 0 else n
            for _ in range(n_prototypes):
                idx = rng.choice(n, size=sample_size, replace=True)
                p = X[idx].mean(axis=0)
                proto_vecs.append(p)
                proto_labels.append(cond)
        else:
            raise ValueError(f"Unknown library_type: {library_type}")

    proto_vecs = np.stack(proto_vecs, axis=0).astype(np.float32) if proto_vecs else np.zeros((0, z_train.shape[1]), dtype=np.float32)

    # normalize prototypes for cosine
    norms = np.linalg.norm(proto_vecs, axis=1, keepdims=True)
    proto_vecs = proto_vecs / np.maximum(norms, 1e-8)

    candidate_conditions = sorted(set(proto_labels))
    return proto_vecs, proto_labels, candidate_conditions


def _retrieve_topk(
    z_query: np.ndarray,
    proto_vecs: np.ndarray,
    proto_labels: List[str],
    candidate_conditions: List[str],
    top_k: List[int],
):
    """
    Retrieve top-K condition predictions for each query using max aggregation across prototypes.
    (Matches the 'prototype max' idea used in cell_eval.) :contentReference[oaicite:11]{index=11}

    Returns:
        predictions: List[List[str]] length N
        all_scores: np.ndarray shape (N, C) scores per condition (after max aggregation)
    """
    if z_query.size == 0 or proto_vecs.size == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    # map condition -> indices of prototypes
    cond_to_proto_idx = defaultdict(list)
    for i, c in enumerate(proto_labels):
        cond_to_proto_idx[c].append(i)

    cond_list = candidate_conditions
    cond_to_idx = {c: i for i, c in enumerate(cond_list)}

    # cosine scores to each prototype
    # z_query: (N, D), proto_vecs: (M, D)
    sim_proto = z_query @ proto_vecs.T  # (N, M)

    # aggregate to condition scores by max over prototypes
    scores = np.full((z_query.shape[0], len(cond_list)), -np.inf, dtype=np.float32)
    for cond, idxs in cond_to_proto_idx.items():
        j = cond_to_idx[cond]
        scores[:, j] = np.max(sim_proto[:, idxs], axis=1)

    max_k = max(top_k)
    top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :max_k]
    predictions = [[cond_list[i] for i in row] for row in top_idx]
    return predictions, scores


def _labels_to_conditions(y: np.ndarray, label2cond: Dict[int, str]) -> List[str]:
    return [label2cond.get(int(v), str(int(v))) for v in y.tolist()]


def save_eval_artifacts(
    *,
    output_dir: Path,
    split_name: str,
    config: dict,
    metrics: dict,
    details: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"eval_{split_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2, ensure_ascii=False)

    details_path = output_dir / f"eval_{split_name}_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    # lightweight markdown report (human-readable)
    # key metrics (aligned with metrics.py + report.py table intent) :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}
    key_fields = [
        "exact_hit@1", "exact_hit@5",
        "relevant_hit@1", "relevant_hit@5",
        "mrr", "ndcg@5",
        "macro_hit@1", "macro_hit@5",
        "n_queries", "n_in_pool", "n_conditions",
    ]
    lines = []
    lines.append(f"# scGPT Finetune Evaluation Report ({split_name})")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    for k in key_fields:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.6f}")
            else:
                lines.append(f"- **{k}**: {v}")
    lines.append("")

    # confidence summary
    if "confidence_auc" in metrics:
        lines.append("## Confidence")
        lines.append("")
        lines.append(f"- **confidence_auc**: {metrics['confidence_auc']:.6f}")
        lines.append("")

    # error analysis
    if "error_analysis" in metrics:
        ea = metrics["error_analysis"]
        lines.append("## Error Analysis")
        lines.append("")
        lines.append(f"- overall_accuracy(top1): {ea.get('overall_accuracy', 0.0):.6f}")
        lines.append(f"- n_queries: {ea.get('n_queries', 0)}")
        lines.append("")
        lines.append("### Most Confused Pairs")
        for p in ea.get("confused_pairs", [])[:10]:
            lines.append(f"- true={p['true']} pred={p['predicted']} count={p['count']}")
        lines.append("")
        lines.append("### Hardest Conditions (Hit@1)")
        for h in ea.get("hardest_conditions", [])[:10]:
            lines.append(f"- {h['condition']}: acc={h['accuracy']:.6f} (n={h['n_queries']})")
        lines.append("")

    report_path = output_dir / f"eval_{split_name}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return metrics_path, details_path, report_path

@torch.no_grad()
def run_full_retrieval_evaluation(
    *,
    model,
    device: str,
    pad_token_id: int,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    label2cond: Dict[int, str],
    top_k: List[int],
    output_dir: Path,
    split_name: str,
    config_dict: dict,
    library_type: str = "mean",
    n_prototypes: int = 30,
    m_samples: int = 50,
    seed: int = 42,
    enable_confidence: bool = False,
    enable_error_analysis: bool = False,
):
    """
    Full evaluation with the SAME metrics as repo eval code:
    - exact_hit@K / relevant_hit@K / mrr / ndcg@K / macro_hit@K ... :contentReference[oaicite:1]{index=1}

    Branch:
    - config.loss_fn == 'classification': rank by classifier logits (closed-set)
    - else: prototype cosine retrieval (open-set-friendly)
    """
    base = model.module if hasattr(model, "module") else model

    # ---------- branch A: classification logits ranking ----------
    if getattr(base.config, "loss_fn", None) == "classification":
        # encode queries
        z_eval, y_eval = _encode_projected_embeddings(model, eval_loader, device, pad_token_id)
        gt_conditions = _labels_to_conditions(y_eval, label2cond)

        if z_eval.size == 0:
            metrics = compute_all_metrics([], [], top_k_values=top_k, candidate_pool=[], include_macro=True)
            details = {"ground_truth": gt_conditions, "predictions": [], "top_k": top_k}
            paths = save_eval_artifacts(output_dir=output_dir, split_name=split_name, config=config_dict, metrics=metrics, details=details)
            return metrics, paths

        # classifier logits: [N, C]
        logits = base.loss_fn.classifier(torch.from_numpy(z_eval).to(device)).detach().cpu().float().numpy()

        # candidates are all training classes 0..C-1 mapped to condition string
        num_conditions = int(base.loss_fn.num_conditions)
        candidate_conditions = [label2cond.get(i, str(i)) for i in range(num_conditions)]

        # topK
        max_k = max(top_k)
        top_idx = np.argsort(logits, axis=1)[:, ::-1][:, :max_k]
        predictions = [[candidate_conditions[i] for i in row] for row in top_idx]

        # metrics
        metrics = compute_all_metrics(
            predictions,
            gt_conditions,
            top_k_values=top_k,
            candidate_pool=candidate_conditions,
            include_macro=True,
        )

        # confidence (optional): feed scores matrix to repo scorer :contentReference[oaicite:2]{index=2}
        if enable_confidence:
            # 你也可以换成 softmax(prob)；margin 对 logits 同样适用
            scorer = ConfidenceScorer(method="margin", top_k_agreement=1)
            confidences = scorer.score_batch(logits)
            is_correct = np.array([(preds[0] == true) for preds, true in zip(predictions, gt_conditions)], dtype=bool)
            metrics["confidence_auc"] = compute_auc_coverage_accuracy(confidences, is_correct)
            cov, acc = coverage_accuracy_curve(confidences, is_correct, n_points=20)
            metrics["coverage_accuracy_curve"] = {"coverage": cov.tolist(), "accuracy": acc.tolist()}

        if enable_error_analysis:
            metrics["error_analysis"] = generate_error_report(
                predictions=predictions,
                ground_truth=gt_conditions,
                k=1,
                n_confused_pairs=10,
                n_hardest=10,
            )

        # details: 保存 logits + softmax 概率（你想要的部分）
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

        details = {
            "ground_truth": gt_conditions,
            "predictions": predictions,
            "top_k": top_k,
            "candidate_conditions": candidate_conditions,
            "scores_type": "classification_logits",
            "topk_label_ids": top_idx.tolist(),
            "topk_logits": np.take_along_axis(logits, top_idx, axis=1).tolist(),
            "topk_probs": np.take_along_axis(probs, top_idx, axis=1).tolist(),
        }

        paths = save_eval_artifacts(
            output_dir=output_dir,
            split_name=split_name,
            config=config_dict,
            metrics=metrics,
            details=details,
        )
        return metrics, paths

    # ---------- branch B: prototype cosine retrieval (InfoNCE / general) ----------
    z_train, y_train = _encode_projected_embeddings(model, train_loader, device, pad_token_id)
    z_eval, y_eval = _encode_projected_embeddings(model, eval_loader, device, pad_token_id)
    gt_conditions = _labels_to_conditions(y_eval, label2cond)

    proto_vecs, proto_labels, candidate_conditions = _build_prototype_library(
        z_train, y_train, label2cond,
        library_type=library_type,
        n_prototypes=n_prototypes,
        m_samples=m_samples,
        seed=seed,
    )
    predictions, scores = _retrieve_topk(z_eval, proto_vecs, proto_labels, candidate_conditions, top_k)

    metrics = compute_all_metrics(
        predictions,
        gt_conditions,
        top_k_values=top_k,
        candidate_pool=candidate_conditions,
        include_macro=True,
    )

    if enable_confidence and scores.size > 0:
        scorer = ConfidenceScorer(method="margin", top_k_agreement=1)
        confidences = scorer.score_batch(scores)
        is_correct = np.array([(preds[0] == true) for preds, true in zip(predictions, gt_conditions)], dtype=bool)
        metrics["confidence_auc"] = compute_auc_coverage_accuracy(confidences, is_correct)
        cov, acc = coverage_accuracy_curve(confidences, is_correct, n_points=20)
        metrics["coverage_accuracy_curve"] = {"coverage": cov.tolist(), "accuracy": acc.tolist()}

    if enable_error_analysis and predictions:
        metrics["error_analysis"] = generate_error_report(
            predictions=predictions,
            ground_truth=gt_conditions,
            k=1,
            n_confused_pairs=10,
            n_hardest=10,
        )

    details = {
        "ground_truth": gt_conditions,
        "predictions": predictions,
        "top_k": top_k,
        "candidate_conditions": candidate_conditions,
        "scores_type": "prototype_cosine",
    }
    paths = save_eval_artifacts(
        output_dir=output_dir,
        split_name=split_name,
        config=config_dict,
        metrics=metrics,
        details=details,
    )
    return metrics, paths


# ----------------------------
# CLI / Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="scGPT Fine-tuning (Tahoe-only)")
    parser.add_argument("--parquet_dir", type=str, default=None)
    parser.add_argument("--finetune_checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--do_binning", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--head_output_dim", type=int, default=None)
    parser.add_argument("--head_hidden_dim", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--infonce_temperature", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--tahoe2scgpt_json", type=str, default=None)
    parser.add_argument("--scgpt_model_dir", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cache_embeddings", action="store_true",
                    help="Precompute and cache CLS embeddings (head_only only).")
    parser.add_argument("--emb_cache_dir", type=str, default=None,
                    help="Where to store cached embeddings (default: checkpoint_dir/emb_cache).")
    parser.add_argument("--max_steps", type=int, default=0, 
                        help="If >0, limit steps per epoch for debugging.")
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
    # ---- full eval args ----
    parser.add_argument(
        "--eval_top_k",
        type=str,
        default="1,5,8,10",
        help="Comma-separated K values for retrieval metrics (default: 1,5,8,10)",
    )
    parser.add_argument(
        "--eval_library",
        type=str,
        choices=["mean", "bootstrap"],
        default="mean",
        help="Prototype library type for retrieval evaluation",
    )
    parser.add_argument(
        "--eval_n_prototypes",
        type=int,
        default=30,
        help="Number of prototypes per class when eval_library=bootstrap",
    )
    parser.add_argument(
        "--eval_m_samples",
        type=int,
        default=50,
        help="Samples per prototype when eval_library=bootstrap",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Seed for bootstrap prototype sampling",
    )
    parser.add_argument(
        "--eval_enable_confidence",
        action="store_true",
        help="Enable confidence AUC + coverage-accuracy curve",
    )
    parser.add_argument(
        "--eval_enable_error_analysis",
        action="store_true",
        help="Enable confusion/hardest-condition report",
    )
    parser.add_argument(
        "--eval_report_dir",
        type=str,
        default=None,
        help="Directory to save eval reports (default: checkpoint_dir/eval_reports)",
    )

    return parser.parse_args()


def _setup_ddp() -> Tuple[bool, str, int]:
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
    args = parse_args()
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
    if args.output_dir:
        config.checkpoint_dir = args.output_dir

    # Extra CLI overrides (no YAML)
    if getattr(args, "epochs", None) is not None:
        config.epochs = args.epochs
    if getattr(args, "batch_size", None) is not None:
        config.batch_size = args.batch_size
    if getattr(args, "learning_rate", None) is not None:
        config.learning_rate = args.learning_rate
    if getattr(args, "weight_decay", None) is not None:
        config.weight_decay = args.weight_decay
    if getattr(args, "warmup_ratio", None) is not None:
        config.warmup_ratio = args.warmup_ratio
    if getattr(args, "early_stopping_patience", None) is not None:
        config.early_stopping_patience = args.early_stopping_patience

    if getattr(args, "scgpt_model_dir", None) is not None:
        config.scgpt_model_dir = args.scgpt_model_dir
    if getattr(args, "tahoe2scgpt_json", None) is not None:
        config.tahoe2scgpt_json = args.tahoe2scgpt_json

    if getattr(args, "label_smoothing", None) is not None:
        config.label_smoothing = args.label_smoothing
    if getattr(args, "infonce_temperature", None) is not None:
        config.infonce_temperature = args.infonce_temperature

    if getattr(args, "lora_rank", None) is not None:
        config.lora_rank = args.lora_rank
    if getattr(args, "lora_alpha", None) is not None:
        config.lora_alpha = args.lora_alpha
    if getattr(args, "lora_dropout", None) is not None:
        config.lora_dropout = args.lora_dropout

    if getattr(args, "head_hidden_dim", None) is not None:
        config.head_hidden_dim = args.head_hidden_dim
    if getattr(args, "head_output_dim", None) is not None:
        config.head_output_dim = args.head_output_dim
    if getattr(args, "head_dropout", None) is not None:
        config.head_dropout = args.head_dropout

    if not config.parquet_dir:
        raise ValueError("Tahoe-only finetune requires --parquet_dir (or data.parquet_dir in YAML).")

    ddp_enabled, device, rank = _setup_ddp()
    is_master = rank == 0

    if is_master:
        print("\n" + "=" * 60)
        print("scGPT Fine-tuning for Reverse Perturbation Retrieval (Tahoe-only)")
        print("=" * 60)
        print(f"Mode: {config.mode}")
        print(f"Loss: {config.loss_fn}")
        print(f"Parquet dir: {config.parquet_dir}")
        print("=" * 60 + "\n")

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
    from scgpt.data_collator import DataCollator

    if is_master:
        print("Loading scGPT model...")
    encoder = ScGPTEncoder(model_dir=config.scgpt_model_dir)
    encoder._load_model()
    scgpt_model = encoder._model

    parquet_dir = config.parquet_dir
    if is_master:
        print(f"Using Tahoe parquet dataset: {parquet_dir}")

    # Detect whether parquet 'genes' are already scGPT vocab ids (pre-remapped in build_dataset.py)
    genes_already_scgpt = False
    id_space_path = os.path.join(parquet_dir, "id_space.json")
    if os.path.isfile(id_space_path):
        try:
            with open(id_space_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            genes_already_scgpt = (str(obj.get("genes", "")).lower() == "scgpt")
        except Exception as e:
            if is_master:
                print(f"[Warn] Failed to parse {id_space_path}: {e}. Will remap Tahoe->scGPT in dataloader.")
            genes_already_scgpt = False
    if is_master:
        print(f"[Info] genes_already_scgpt = {genes_already_scgpt}")

    # 1) label vocab -> num_conditions, label2gene
    vocab_path = os.path.join(parquet_dir, "label_vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_obj = json.load(f)

    gene2label = vocab_obj.get("gene2label") or vocab_obj.get("gene2y")
    if gene2label is None:
        raise ValueError(f"label_vocab.json missing gene2label/gene2y: {vocab_path}")
    num_conditions = len(gene2label)

    label2gene_raw = vocab_obj.get("label2gene") or vocab_obj.get("y2gene")
    if label2gene_raw is None:
        # build from gene2label if needed
        label2gene = {int(v): str(k) for k, v in gene2label.items()}
    else:
        label2gene = {int(k): v for k, v in label2gene_raw.items()}

    # 2) build label -> target gene scGPT id for masking
    token_vocab = encoder._vocab  # GeneVocab supports token lookup
    label2target_scgptid: Dict[int, int] = {}
    for lab, g in label2gene.items():
        if g in token_vocab:
            label2target_scgptid[int(lab)] = int(token_vocab[g])

    # 3) split parquet paths
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

    # 4) model
    model = FineTunableScGPTEncoder(
        scgpt_model=scgpt_model,
        config=config,
        num_conditions=num_conditions,
    )

    # 5) datasets
    cls_id = int(encoder._vocab["<cls>"])
    pad_token_id = int(encoder._vocab[encoder._model_configs["pad_token"]])
    pad_value = float(encoder._model_configs["pad_value"])

    train_subset = ParquetTokenDataset(
        train_paths,
        cls_token_id=cls_id,
        pad_value=pad_value,
        pad_token_id=pad_token_id,
        tahoe2scgpt_json=config.tahoe2scgpt_json,
        remap_tahoe_to_scgpt=(not genes_already_scgpt),
        label2target_scgptid=label2target_scgptid,
        mask_value=0.0,
    )
    val_subset = ParquetTokenDataset(
        val_paths,
        cls_token_id=cls_id,
        pad_value=pad_value,
        pad_token_id=pad_token_id,
        tahoe2scgpt_json=config.tahoe2scgpt_json,
        remap_tahoe_to_scgpt=(not genes_already_scgpt),
        label2target_scgptid=label2target_scgptid,
        mask_value=0.0,
    )
    test_dataset = ParquetTokenDataset(
        test_paths,
        cls_token_id=cls_id,
        pad_value=pad_value,
        pad_token_id=pad_token_id,
        tahoe2scgpt_json=config.tahoe2scgpt_json,
        remap_tahoe_to_scgpt=(not genes_already_scgpt),
        label2target_scgptid=label2target_scgptid,
        mask_value=0.0,
    )
    ood_dataset = None
    if len(ood_paths) > 0:
        ood_dataset = ParquetTokenDataset(
            ood_paths,
            cls_token_id=cls_id,
            pad_value=pad_value,
            pad_token_id=pad_token_id,
            tahoe2scgpt_json=config.tahoe2scgpt_json,
            label2target_scgptid=label2target_scgptid,
            mask_value=0.0,
        )

    # 6) collator
    collator = DataCollator(
        do_padding=True,
        pad_token_id=pad_token_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=config.do_binning,  # default False for log1p
        max_length=encoder.max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    collate_fn = _make_scgpt_collate_fn(collator)

    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_sampler = torch.utils.data.DistributedSampler(train_subset, shuffle=False)
        val_sampler = torch.utils.data.DistributedSampler(val_subset, shuffle=False)

    loader_kwargs = dict(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
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

    # ---- quick sanity check: batch size & token stats ----
    if is_master:
        print("\n" + "-" * 60)
        print("[Sanity] DataLoader / batch stats")
        print(f"  config.batch_size = {config.batch_size}")
        print(f"  len(train_loader) = {len(train_loader)}  (steps per epoch)")
        print(f"  len(val_loader)   = {len(val_loader)}")
        print(f"  pad_token_id      = {pad_token_id}")
        print("-" * 60)

        # Peek a few batches to estimate padding ratio and effective tokens
        peek_batches = 5
        total_tokens = 0
        total_nonpad = 0
        seen = 0

        for i, b in enumerate(train_loader):
            gene = b["gene"]  # [B, L]
            # total tokens in batch
            tok = gene.numel()
            # non-pad tokens in batch
            nonpad = gene.ne(pad_token_id).sum().item()

            total_tokens += tok
            total_nonpad += nonpad
            seen += 1

            if i < 3:
                # print first 3 batch shapes for confirmation
                print(f"[Sanity] batch {i}: gene.shape={tuple(gene.shape)} "
                      f"tokens={tok} nonpad={nonpad} pad_ratio={1 - (nonpad / max(tok,1)):.4%}")

            if seen >= peek_batches:
                break

        if seen > 0:
            avg_pad_ratio = 1 - (total_nonpad / max(total_tokens, 1))
            avg_nonpad_per_batch = total_nonpad / seen
            print(f"[Sanity] avg over {seen} batches: "
                  f"avg_nonpad_tokens_per_batch={avg_nonpad_per_batch:.1f}, "
                  f"avg_pad_ratio={avg_pad_ratio:.4%}")
        print("-" * 60 + "\n")

    # ---- parse eval args ----
    eval_top_k = [int(x) for x in args.eval_top_k.split(",") if x.strip()]
    eval_report_dir = Path(args.eval_report_dir) if args.eval_report_dir \
            else Path(config.checkpoint_dir) / "eval_reports"

    # DDP wrap
    if ddp_enabled:
        allow_unused = config.loss_fn == "infonce" or config.mode == "lora_head"
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[int(device.split(":")[-1])],
            find_unused_parameters=allow_unused,
        )

    end_to_end_flag = True  # Tahoe-only path is end-to-end token -> cls -> head
    trainer = ScGPTTrainer(
        model,
        config,
        device=device,
        is_master=is_master,
        end_to_end=end_to_end_flag,
        pad_token_id=pad_token_id,
    )
    if args.cache_embeddings:
        if config.mode != "head_only":
            raise ValueError("--cache_embeddings is intended for head_only mode only.")

        emb_dir = Path(args.emb_cache_dir) if args.emb_cache_dir else Path(config.checkpoint_dir) / "emb_cache"
        # 先把 backbone+head 加载到位（如果你要从某个初始 checkpoint 开始也可以）
        # 然后缓存 train/val
        build_embedding_cache(model=trainer.model, device=device, pad_token_id=pad_token_id,
                            dataset=train_subset, collate_fn=collate_fn, split_name="train", out_dir=emb_dir,
                            batch_size=config.batch_size)
        build_embedding_cache(model=trainer.model, device=device, pad_token_id=pad_token_id,
                            dataset=val_subset, collate_fn=collate_fn, split_name="val", out_dir=emb_dir,
                            batch_size=config.batch_size)

        # 用 embedding dataset 替换 train_loader / val_loader
        train_z = emb_dir / "train_z.npy"
        train_y = emb_dir / "train_y.npy"
        val_z = emb_dir / "val_z.npy"
        val_y = emb_dir / "val_y.npy"

        train_loader = DataLoader(EmbeddingDataset(train_z, train_y), batch_size=config.batch_size,
                                shuffle=True, num_workers=2, pin_memory=True, collate_fn=emb_collate)
        val_loader = DataLoader(EmbeddingDataset(val_z, val_y), batch_size=config.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True, collate_fn=emb_collate)

        print("[Cache] Switched training to cached embeddings. Backbone forward will be skipped.")

    # Eval-only / frozen
    if config.mode == "frozen" or config.eval_only:
        if not config.finetune_checkpoint:
            raise ValueError("frozen/eval_only requires --finetune_checkpoint to load head/LoRA weights")
        trainer.load_checkpoint(config.finetune_checkpoint)

        # loaders for eval
        train_eval_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # label -> condition string (gene name)
        label2cond = {int(k): str(v) for k, v in label2gene.items()}

        if is_master:
            print("[Eval] Running full retrieval evaluation on TEST...")

        metrics_test, paths_test = run_full_retrieval_evaluation(
            model=trainer.model,
            device=device,
            pad_token_id=pad_token_id,
            train_loader=train_eval_loader,
            eval_loader=test_loader,
            label2cond=label2cond,
            top_k=eval_top_k,
            output_dir=eval_report_dir,
            split_name="test",
            config_dict=asdict(config),
            library_type=args.eval_library,
            n_prototypes=args.eval_n_prototypes,
            m_samples=args.eval_m_samples,
            seed=args.eval_seed,
            enable_confidence=args.eval_enable_confidence,
            enable_error_analysis=args.eval_enable_error_analysis,
        )

        if is_master:
            print(f"[Eval][TEST] exact_hit@1={metrics_test.get('exact_hit@1', 0.0):.4f}")
            print(f"[Eval][TEST] report saved: {paths_test[-1]}")

        # OOD (optional)
        if ood_dataset is not None:
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            if is_master:
                print("[Eval] Running full retrieval evaluation on OOD...")

            metrics_ood, paths_ood = run_full_retrieval_evaluation(
                model=trainer.model,
                device=device,
                pad_token_id=pad_token_id,
                train_loader=train_eval_loader,
                eval_loader=ood_loader,
                label2cond=label2cond,
                top_k=eval_top_k,
                output_dir=eval_report_dir,
                split_name="ood",
                config_dict=asdict(config),
                library_type=args.eval_library,
                n_prototypes=args.eval_n_prototypes,
                m_samples=args.eval_m_samples,
                seed=args.eval_seed,
                enable_confidence=args.eval_enable_confidence,
                enable_error_analysis=args.eval_enable_error_analysis,
            )
            if is_master:
                print(f"[Eval][OOD] exact_hit@1={metrics_ood.get('exact_hit@1', 0.0):.4f}")
                print(f"[Eval][OOD] report saved: {paths_ood[-1]}")
        return


    # Train
    history = trainer.train(train_loader, val_loader)
    if is_master:
        # run full eval after training
        eval_top_k = [int(x) for x in args.eval_top_k.split(",") if x.strip()]
        eval_report_dir = Path(args.eval_report_dir) if args.eval_report_dir else Path(config.checkpoint_dir) / "eval_reports"

        train_eval_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        label2cond = {int(k): str(v) for k, v in label2gene.items()}

        print("[Eval] Post-train full retrieval evaluation on TEST...")
        run_full_retrieval_evaluation(
            model=trainer.model,
            device=device,
            pad_token_id=pad_token_id,
            train_loader=train_eval_loader,
            eval_loader=test_loader,
            label2cond=label2cond,
            top_k=eval_top_k,
            output_dir=eval_report_dir,
            split_name="test",
            config_dict=asdict(config),
            library_type=args.eval_library,
            n_prototypes=args.eval_n_prototypes,
            m_samples=args.eval_m_samples,
            seed=args.eval_seed,
            enable_confidence=args.eval_enable_confidence,
            enable_error_analysis=args.eval_enable_error_analysis,
        )


    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
