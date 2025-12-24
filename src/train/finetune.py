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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

        with open(tahoe2scgpt_json, "r", encoding="utf-8") as f:
            self.tahoe_gene_map = {int(k): int(v) for k, v in json.load(f).items()}

        # Tahoe special tokens are typically: <pad>=0, <cls>=1, <eoc>=2
        self.tahoe_special_map = {
            0: self.pad_token_id,
            1: self.cls_token_id,
        }
        if self.eoc_token_id is not None:
            self.tahoe_special_map[2] = self.eoc_token_id

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

        for batch in dataloader:
            input_gene_ids = batch["gene"].to(self.device)
            expressions = batch["expr"].to(self.device)
            labels = batch["labels"].to(self.device)

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

            self.optimizer.zero_grad()
            loss = base_model.compute_loss(projected, labels)
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

        from tqdm import tqdm

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
        label2target_scgptid=label2target_scgptid,
        mask_value=0.0,
    )
    val_subset = ParquetTokenDataset(
        val_paths,
        cls_token_id=cls_id,
        pad_value=pad_value,
        pad_token_id=pad_token_id,
        tahoe2scgpt_json=config.tahoe2scgpt_json,
        label2target_scgptid=label2target_scgptid,
        mask_value=0.0,
    )
    test_dataset = ParquetTokenDataset(
        test_paths,
        cls_token_id=cls_id,
        pad_value=pad_value,
        pad_token_id=pad_token_id,
        tahoe2scgpt_json=config.tahoe2scgpt_json,
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

    # Eval-only / frozen
    if config.mode == "frozen" or config.eval_only:
        if not config.finetune_checkpoint:
            raise ValueError("frozen/eval_only requires --finetune_checkpoint to load head/LoRA weights")
        trainer.load_checkpoint(config.finetune_checkpoint)

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

    # Train
    history = trainer.train(train_loader, val_loader)
    if is_master:
        history_path = Path(config.checkpoint_dir) / f"history_{config.mode}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"Training history saved: {history_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
