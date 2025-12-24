# VCC - Reverse Perturbation Prediction

Reverse perturbation prediction for CRISPR Perturb-seq data using retrieval-based/classification methods.

## Quick Start

```bash
conda create -n vcc python=3.11 -y
conda activate vcc
pip install -r requirements.txt

python -m src.main --config src/configs/pca.yaml
```

## Repository Layout

```
.
├── src/                # Core pipeline package
│   ├── configs/        # Experiment configuration files
│   ├── data/           # Dataset loading and split logic
│   ├── evaluate/       # Evaluation metrics and helpers
│   ├── model/          # Encoders and retrieval models
│   ├── train/          # Training and fine-tuning flows
│   └── utils/          # Shared utilities
├── scripts/            # Automation helpers and SLURM runners
├── scGPT/              # Vendorized scGPT modules and tests
├── data/
│   ├── raw/            # Raw inputs (gitignored)
│   └── processed/      # Derived features
├── docs/               # Project documentation and references
├── tests/              # Project tests
└── cell-eval/          # Standalone evaluation package
```

## Documentation

- Data splits and AnnData requirements: `docs/data.md`
- Metrics and evaluation: `docs/eval_metrics.md`
- Project overview: `docs/project-intro/introduction.md`
- scGPT reference notes: `docs/references/scGPT.md`

## Tahoe dataset
```bash
pip show scgpt 
```
Name: scgpt Version: 0.2.0

- for head only model
```bash
# train
python -m src.train.finetune \
  --mode head_only \
  --loss classification \
  --parquet_dir ./tahoe/tahoe_scgpt_single_target_log1p \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --epochs 20 \
  --cache_embeddings \
  --emb_cache_dir model/scgpt_finetune/emb_cache_headonly_cls

# eval: use this is ok
python -m src.train.finetune \
  --mode head_only \
  --loss classification \
  --parquet_dir ./tahoe/tahoe_scgpt_single_target_log1p \
  --finetune_checkpoint model/scgpt_finetune/best_head_only.pt \
  --eval_only \
  --eval_enable_confidence \
  --eval_enable_error_analysis

# eval: if you want more setting
python -m src.train.finetune \
  --mode head_only \
  --loss classification \
  --parquet_dir /tahoe/tahoe_scgpt_single_target_log1p \
  --finetune_checkpoint model/scgpt_finetune/best_head_only.pt \
  --eval_only \
  --eval_top_k 1,5,10 \
  --eval_report_dir model/scgpt_finetune/eval_reports_head_only \
  --eval_enable_confidence \
  --eval_enable_error_analysis

```

- for lora head model
```bash
# train
python -m src.train.finetune \
  --mode lora_head \
  --loss classification \
  --parquet_dir /tahoe/tahoe_scgpt_single_target_log1p \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --epochs 10

# eval
python -m src.train.finetune \
  --mode lora_head \
  --loss classification \
  --parquet_dir ./tahoe/tahoe_scgpt_single_target_log1p \
  --finetune_checkpoint model/scgpt_finetune/best_lora_head.pt \
  --eval_only \
  --eval_enable_confidence \
  --eval_enable_error_analysis

python -m src.train.finetune \
  --mode lora_head \
  --loss classification \
  --parquet_dir ./tahoe/tahoe_scgpt_single_target_log1p \
  --finetune_checkpoint model/scgpt_finetune/best_lora_head.pt \
  --eval_only \
  --eval_top_k 1,5,10 \
  --eval_report_dir model/scgpt_finetune/eval_reports_lora_head \
  --eval_enable_confidence \
  --eval_enable_error_analysis

```

[Eval] test acc: 0.0475
[Eval] ood test acc: 0.0447

- if you want to record time
```bash
python -m src.train.finetune \
  --mode head_only \
  --loss classification \
  --parquet_dir ./tahoe/tahoe_scgpt_single_target_log1p \
  --epochs 10 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --early_stopping_patience 5 \
  --scgpt_model_dir model/scGPT \
  --tahoe2scgpt_json ./tahoe/tahoe_tokenid_to_scgptid.json
  --max_steps 200

# after adding cache
python -m src.train.finetune \
  --mode head_only \
  --loss classification \
  --parquet_dir ./tahoe/tahoe_scgpt_single_target_log1p \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --epochs 10 \
  --cache_embeddings \
  --max_steps 200

```