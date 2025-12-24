# tahoe data

## process
- 只保留单靶点药物
- train/test/val split: 0.9*(0.8,0.1,0.1)
- OOD test: 0.1
- 表达值做 log1p + Top-K 基因筛选，只保留<=2048个表达最高的基因
- 不符合要求的数据全部丢弃

```python
{
  "genes":         List[int],
  "expressions":   List[float],
  "label":         int,
  "target_gene":   str,
  "drug":          str,
  "cell_line_id":  str,
  "sample":        str,
  "plate":         str,
  "split":         "train" | "val" | "test" | "ood_test"
}
```

## take a look
- `label_vocab.json` 基因id到基因名称的映射
- 数据分割

```bash
OUT_DIR: ./tahoe_scgpt_single_target_log1p
[label_vocab.json] keys: ['gene2label', 'label2gene']
  num_classes: 94
  gene2label preview: { 
    "gene2label": {
        "ABCB1": 0,
        "ABL1": 1,
        "ACE": 2,
        "ADRB1": 3，...}

[Parquet files] total: 53
  ood: 4 files
  test: 5 files
  train: 39 files
  val: 5 files
```
- train data 
```bash
Total rows: 50000
genes: list<element: int64>
  child 0, element: int64
expressions: list<element: double>
  child 0, element: double
label: int64
target_gene: string
drug: string
cell_line_id: string
sample: string
plate: string
split: string
```
- first row 
```bash
split: train
label: 20
target_gene: CACNA1C
drug: Berbamine
cell_line_id: CVCL_1550
sample: smp_1786
plate: plate4

Genes length: 1556 # different length for different genes
Expressions length: 1556

First 20 genes:
[ 14  19  20  27  32  38  45  78  84 100 103 104 109 112 114 137 149 171
 187 202]

First 20 expressions: # log1p
[0.69314718 0.69314718 0.69314718 0.69314718 0.69314718 0.69314718
 0.69314718 0.69314718 0.69314718 0.69314718 0.69314718 0.69314718
 0.69314718 1.09861231 0.69314718 1.09861231 1.09861231 0.69314718
 0.69314718 1.09861231]

Last 20 genes:
[50527 51573 51861 52273 52970 52993 53130 53602 54159 54621 54998 57680
 58170 60700 60753 60761 61490 61807 62276 62614]

Last 20 expressions:
[0.69314718 0.69314718 0.69314718 0.69314718 0.69314718 0.69314718
 1.09861231 0.69314718 1.38629436 0.69314718 0.69314718 1.38629436
 0.69314718 0.69314718 0.69314718 0.69314718 1.09861231 0.69314718
 0.69314718 0.69314718]
```

- ood data
```bash
==================== ood ====================

File[0]: ./tahoe_scgpt_single_target_log1p/ood_test_0000.parquet
--- ood sample from ood_test_0000.parquet: showing 3 rows ---

[Row 0] split=ood_test label=25 target_gene=COMT drug=Tolcapone
        cell_line_id=CVCL_0179 sample=smp_1813 plate=plate4

[Row 1] split=ood_test label=75 target_gene=REN drug=Aliskiren
        cell_line_id=CVCL_0179 sample=smp_1815 plate=plate4

[Row 2] split=ood_test label=55 target_gene=MTOR drug=Temsirolimus
        cell_line_id=CVCL_0028 sample=smp_1802 plate=plate4

```