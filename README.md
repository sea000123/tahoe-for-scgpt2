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

```bash
python -m src.train.finetune \
  --mode head_only --loss classification \
  --parquet_dir /home/user/Desktop/CODE/VCC/Tahoe/raw/tahoe_scgpt_single_target_log1p

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

python -m src.train.finetune \
  --mode lora_head --loss classification \
  --parquet_dir /tahoe/tahoe_scgpt_single_target_log1p

python -m src.train.finetune \
  --mode frozen --loss classification --eval_only \
  --parquet_dir /home/user/Desktop/CODE/VCC/Tahoe/raw/tahoe_scgpt_single_target_log1p \
  --finetune_checkpoint './model/scgpt_finetune/final_head_only.pt'

```

[Eval] test acc: 0.0475
[Eval] ood test acc: 0.0447
