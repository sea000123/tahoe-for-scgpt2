import os, glob, json, re, hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
LOCAL_TAHOE_DIR = "./tahoe_small_download"
DATA_GLOB = os.path.join(LOCAL_TAHOE_DIR, "data", "train-*.parquet")

OUT_DIR = "./tahoe_scgpt_single_target_log1p"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 286
OOD_CELL_LINE_FRAC = 0.1  # 1/10 cell lines held out for OOD generalization

# in-domain row split (by BARCODE_SUB_LIB_ID stable hash)
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

TOP_K_GENES = 2048
ROWS_PER_SHARD = 50_000
BATCH_SIZE = 4096

# optional: exclude DMSO-like drugs if present
CTRL_REGEX = re.compile(r"\bDMSO\b", re.IGNORECASE)

parquet_write_kwargs = dict(compression="zstd", compression_level=3)

# output columns aligned to your training code
KEEP_COLS = [
    "genes",         # List[int]  (NO <cls> here)
    "expressions",   # List[float] (NO pad_value here)
    "label",         # int
    "target_gene",   # str
    "drug",          # str
    "cell_line_id",  # str
    "sample",        # str
    "plate",         # str
    "split",         # str in {"train","val","test","ood_test"}
]

# ----------------------------
# Helpers
# ----------------------------
_SPLIT_RE = re.compile(r"[,\;\|]|\s+")

def parse_targets(x):
    """Parse targets to list[str]."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        xs = x
    elif isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "none":
            return []
        xs = [t.strip() for t in _SPLIT_RE.split(s) if t.strip()]
    else:
        return []
    # dedup keep order
    seen, out = set(), []
    for t in xs:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def is_control_drug(drug: str) -> bool:
    return bool(drug) and bool(CTRL_REGEX.search(drug))

def stable_u01(s: str) -> float:
    """Deterministic uniform(0,1) from string."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    v = int(h[:16], 16)
    return (v % 10_000_000) / 10_000_000.0

def clean_log1p_topk(genes, exprs, topk=TOP_K_GENES):
    """
    - drop marker token at position 0
    - clip exprs to >=0 then log1p
    - keep top-k by value (log1p is non-negative)
    """
    if genes is None or exprs is None:
        return [], []
    if len(genes) == 0 or len(exprs) == 0 or len(genes) != len(exprs):
        return [], []

    genes = genes[1:]
    exprs = exprs[1:]
    if len(genes) == 0:
        return [], []

    g = np.asarray(genes, dtype=np.int64)
    x = np.asarray(exprs, dtype=np.float32)

    # important: avoid NaN from negative values
    x = np.maximum(x, 0.0)
    x = np.log1p(x)

    if len(g) > topk:
        idx = np.argpartition(x, -topk)[-topk:]
        g = g[idx]
        x = x[idx]

    return g.tolist(), x.astype(np.float32).tolist()

# ----------------------------
# 0) drug_metadata: keep only single-target drugs
# ----------------------------
drug_md_path = os.path.join(LOCAL_TAHOE_DIR, "metadata", "drug_metadata.parquet")
drug_df = pq.read_table(drug_md_path).to_pandas()
drug_df["targets_list"] = drug_df["targets"].apply(parse_targets)

drug_df = drug_df[drug_df["targets_list"].map(len) == 1].copy()
drug_df["target_gene"] = drug_df["targets_list"].map(lambda xs: xs[0])

# optional: remove DMSO in metadata
drug_df = drug_df[~drug_df["drug"].astype(str).str.contains("DMSO", case=False, na=False)].copy()

drug2target = dict(zip(drug_df["drug"], drug_df["target_gene"]))
targets_sorted = sorted(set(drug2target.values()))
gene2label = {g: i for i, g in enumerate(targets_sorted)}
label2gene = {i: g for g, i in gene2label.items()}

print(f"[Info] Single-target drugs kept: {len(drug2target)}")
print(f"[Info] Classes(target genes) kept: {len(gene2label)}")

with open(os.path.join(OUT_DIR, "label_vocab.json"), "w", encoding="utf-8") as f:
    json.dump({"gene2label": gene2label, "label2gene": label2gene}, f, ensure_ascii=False, indent=2)

# ----------------------------
# 1) Collect cell_line_ids (Pass1)
# ----------------------------
files = sorted(glob.glob(DATA_GLOB))
if not files:
    raise FileNotFoundError(f"No parquet files matched {DATA_GLOB}")

dataset = ds.dataset(files, format="parquet")

scanner = dataset.scanner(columns=["drug", "cell_line_id"], batch_size=BATCH_SIZE)
cell_lines = set()

pbar = tqdm(desc="Pass1: collect cell_line_ids", total=None)
for batch in scanner.to_batches():
    drugs = batch.column(batch.schema.get_field_index("drug")).to_pylist()
    clids = batch.column(batch.schema.get_field_index("cell_line_id")).to_pylist()
    for d, c in zip(drugs, clids):
        if d in drug2target and c is not None and (not is_control_drug(d)):
            cell_lines.add(str(c))
    pbar.update(batch.num_rows)
pbar.close()

cell_lines = sorted(cell_lines)
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(cell_lines)

n_total = len(cell_lines)
n_ood = int(round(n_total * OOD_CELL_LINE_FRAC))
ood_cell_lines = set(cell_lines[:n_ood])
ind_cell_lines = set(cell_lines[n_ood:])

print(f"[Info] Total cell lines (eligible): {n_total}")
print(f"[Info] OOD cell lines held-out: {len(ood_cell_lines)}")
print(f"[Info] In-domain cell lines: {len(ind_cell_lines)}")

with open(os.path.join(OUT_DIR, "cell_line_split.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "random_seed": RANDOM_SEED,
            "ood_frac": OOD_CELL_LINE_FRAC,
            "ood_cell_lines": sorted(list(ood_cell_lines)),
            "in_domain_cell_lines": sorted(list(ind_cell_lines)),
            "in_domain_row_split": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

def assign_split(cell_line_id: str, barcode_sub_lib_id: str) -> str:
    if cell_line_id in ood_cell_lines:
        return "ood_test"
    key = barcode_sub_lib_id if barcode_sub_lib_id else cell_line_id
    u = stable_u01(str(key))
    if u < TRAIN_FRAC:
        return "train"
    elif u < TRAIN_FRAC + VAL_FRAC:
        return "val"
    else:
        return "test"

# ----------------------------
# 2) Pass2: write aligned parquet
# ----------------------------
COLUMNS = ["genes", "expressions", "drug", "sample", "BARCODE_SUB_LIB_ID", "cell_line_id", "plate"]

buffers = {"train": [], "val": [], "test": [], "ood_test": []}
shard_idx = defaultdict(int)

def flush(split_name: str):
    buf = buffers[split_name]
    if not buf:
        return
    table = pa.Table.from_pylist(buf)
    out_path = os.path.join(OUT_DIR, f"{split_name}_{shard_idx[split_name]:04d}.parquet")
    pq.write_table(table, out_path, **parquet_write_kwargs)
    shard_idx[split_name] += 1
    buffers[split_name] = []

scanner = dataset.scanner(columns=COLUMNS, batch_size=BATCH_SIZE)
pbar = tqdm(desc="Pass2: write aligned parquet", total=None)

for batch in scanner.to_batches():
    drugs = batch.column(batch.schema.get_field_index("drug")).to_pylist()
    genes_col = batch.column(batch.schema.get_field_index("genes")).to_pylist()
    exprs_col = batch.column(batch.schema.get_field_index("expressions")).to_pylist()

    sample_col = batch.column(batch.schema.get_field_index("sample")).to_pylist()
    bsl_col = batch.column(batch.schema.get_field_index("BARCODE_SUB_LIB_ID")).to_pylist()
    clid_col = batch.column(batch.schema.get_field_index("cell_line_id")).to_pylist()
    plate_col = batch.column(batch.schema.get_field_index("plate")).to_pylist()

    for i, drug in enumerate(drugs):
        if drug not in drug2target:
            continue
        if is_control_drug(drug):
            continue

        cell_line_id = str(clid_col[i]) if clid_col[i] is not None else None
        if cell_line_id is None:
            continue

        target_gene = drug2target[drug]
        if target_gene not in gene2label:
            continue
        label = int(gene2label[target_gene])

        gene_ids, values = clean_log1p_topk(genes_col[i], exprs_col[i], topk=TOP_K_GENES)
        if len(gene_ids) == 0 or len(gene_ids) != len(values):
            continue

        split = assign_split(cell_line_id, str(bsl_col[i]) if bsl_col[i] is not None else "")

        row = {
            "genes": gene_ids,                 # List[int] (no <cls>)
            "expressions": values,             # List[float] (no pad_value)
            "label": label,                    # int
            "target_gene": target_gene,
            "drug": drug,
            "cell_line_id": cell_line_id,
            "sample": sample_col[i],
            "plate": plate_col[i],
            "split": split,
        }
        buffers[split].append({k: row[k] for k in KEEP_COLS})

        if len(buffers[split]) >= ROWS_PER_SHARD:
            flush(split)

    pbar.update(batch.num_rows)

pbar.close()
for sp in list(buffers.keys()):
    flush(sp)

print("[Done]")
print("Output:", OUT_DIR)
print("Shards:", dict(shard_idx))


'''
export http_proxy=http://192.168.10.108:10808
export https_proxy=http://192.168.10.108:10808
huggingface-cli login

OUT=./tahoe_small_download
mkdir -p "$OUT"

hf download tahoebio/Tahoe-100M \
  --repo-type dataset \
  --local-dir "$OUT" \
  --include "metadata/drug_metadata.parquet"

hf download tahoebio/Tahoe-100M \
  --repo-type dataset \
  --local-dir "$OUT" \
  --include "data/train-00[0-3][0-9][0-9]-of-03388.parquet"
  
hf download tahoebio/Tahoe-100M \
  --repo-type dataset \
  --local-dir "./tahoe_small_download" \
  --include "metadata/gene_metadata.parquet"

预计运行90min
'''