# Cross-dataset generalization in HCC tumour segmentation

Code accompanying the manuscript:

> **Architecture Does Not Determine Cross-Dataset Generalization in Hepatocellular Carcinoma Tumor Segmentation: A Three-Family Comparison with Failure-Mode and Calibration Analysis**
> *(under review)*

Three segmentation architectures — convolutional (nnU-Net ResEnc-M), state-space (U-Mamba Bot) and transformer (SwinUNETR) — are trained on a single HCC cohort under identical splits and applied without adaptation to two independent external cohorts. The repository reproduces every number, table and figure in the paper from public data.

---

## Summary of findings

| | ResEnc-M (CNN) | U-Mamba (SSM) | SwinUNETR (Transformer) |
|---|---|---|---|
| Internal 5-fold Dice | 0.685 ± 0.065 | 0.662 ± 0.044 | 0.479 ± 0.092 |
| 3D-IRCADb (n = 15) | 0.260 | 0.272 | 0.255 |
| LiTS (n = 118) | 0.366 | 0.354 | 0.222 |

Internal performance differs; external performance converges. The architectures fail on the same cases (Spearman ρ = 0.836–0.964), lesion volume governs performance in all six architecture × cohort combinations, and neither anatomical constraint nor probability ensembling recovers the deficit.

---

## Data

All three datasets are public and must be obtained from their original sources; none is redistributed here.

| Cohort | Role | Source |
|---|---|---|
| HCC-TACE-Seg | development (n = 103) | The Cancer Imaging Archive |
| 3D-IRCADb-01 | external (n = 15) | IRCAD, Strasbourg |
| LiTS / MSD Task03 Liver | external (n = 118) | Medical Segmentation Decathlon |

`splits_final.json` (the patient-disjoint five-fold assignment used identically by all three architectures) is included so that the internal cross-validation is exactly reproducible.

---

## Layout

```
data_preparation/
  download_hcc_tace_seg.py     fetch the development cohort from TCIA
  download_needed_ct.py        fetch referenced CT series
  prepare_nnunet_dataset.py    DICOM SEG -> nnU-Net; SOP-based slice alignment
  seg_inspect.py               inspect DICOM SEG geometry
  convert_ircadb.py            3D-IRCADb -> nnU-Net format
  convert_lits.py              MSD Task03 -> nnU-Net format (tumour-bearing cases only)

training/
  nnUNetTrainerUMambaBot_stable.py     AMP off, lr 1e-3, 250 epochs, grad clip 12
  nnUNetTrainerSwinUNETR_250epochs.py  reference SwinUNETR trainer, budget-matched
  splits_final.json                    shared five-fold assignment

evaluation/
  eval_external.py       Dice, IoU, HD95, ASSD with bootstrap CIs
  compare3.py            case-level concordance, volume stratification
  calibration.py         ECE, MCE, reliability curves, failure-time confidence
  selective.py           risk-coverage curves
  extras.py              threshold transfer, cohort characteristics
  liver_constraint.py    anatomical constraint (upper bound)
  ensemble.py            pairwise and three-way probability ensembles

figures/
  figures.py             Figures 1-5
  fig6.py                Figure 6 (qualitative cases)
```

---

## Reproducing the results

Two environments are required, because the ResEnc-M plans need a newer nnU-Net than the U-Mamba fork provides.

```bash
# environment A — ResEnc-M
conda create -n hccseg python=3.10 && conda activate hccseg
pip install nnunetv2                     # 2.4.x

# environment B — U-Mamba and SwinUNETR
conda create -n umamba python=3.10 && conda activate umamba
# follow the U-Mamba installation instructions (nnU-Net 2.2 fork), then:
pip install monai==1.3.0
```

Set the nnU-Net paths in both environments:

```bash
export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUNet_results"
export nnUNet_compile=0
```

### 1. Prepare the data

```bash
python data_preparation/download_hcc_tace_seg.py
python data_preparation/prepare_nnunet_dataset.py     # -> Dataset501_HCCtumor
python data_preparation/convert_ircadb.py --root 3Dircadb1     --out ircadb_nnunet
python data_preparation/convert_lits.py   --root Task03_Liver  --out lits_nnunet
```

Copy `training/splits_final.json` into `nnUNet_preprocessed/Dataset501_HCCtumor/` before training so that all three architectures use the same folds.

### 2. Train

```bash
# ResEnc-M (environment A)
for f in 0 1 2 3 4; do
  nnUNetv2_train Dataset501_HCCtumor 3d_fullres $f \
    -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
done

# U-Mamba and SwinUNETR (environment B) — copy the trainers into
# nnunetv2/training/nnUNetTrainer/ first
for f in 0 1 2 3 4; do
  nnUNetv2_train Dataset501_HCCtumor 3d_fullres $f -tr nnUNetTrainerUMambaBot_stable
done
for f in 0 1 2 3 4; do
  nnUNetv2_train Dataset501_HCCtumor 3d_fullres $f -tr nnUNetTrainerSwinUNETR_250epochs
done
```

Approximately 14 h per fold per architecture on an NVIDIA RTX 6000 Ada (48 GB).

### 3. External inference

Five-fold ensemble with `--save_probabilities` (the probability maps are needed for the calibration, selective-prediction and ensemble analyses):

```bash
nnUNetv2_predict -i ircadb_nnunet/imagesTs -o ircadb_predictions_prob \
  -d Dataset501_HCCtumor -c 3d_fullres \
  -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans \
  -f 0 1 2 3 4 --save_probabilities
```

and equivalently for LiTS and for the other two trainers.

### 4. Analysis

```bash
python evaluation/eval_external.py --pred <pred_dir> --gt <gt_dir> --internal_dice <value>
python evaluation/compare3.py
python evaluation/calibration.py
python evaluation/selective.py
python evaluation/extras.py
python evaluation/liver_constraint.py --lits_root Task03_Liver
python evaluation/ensemble.py
python figures/figures.py
python figures/fig6.py
```

---

## Notes

**Two environments.** ResEnc-M was trained under nnU-Net 2.4-series; U-Mamba and SwinUNETR under the nnU-Net 2.2-based U-Mamba fork. Running a ResEnc plan under the older fork raises `KeyError: 'conv_kernel_sizes'`.

**Mixed precision.** U-Mamba produces non-finite losses with AMP enabled, apparently from float16 overflow in the selective-scan recurrence. The `_stable` trainer disables AMP, lowers the learning rate to 1e-3 and clips gradients at 12.

**Two definitions of complete failure.** `eval_external.py` counts empty predictions; the per-case CSVs count zero overlap. They differ substantially for SwinUNETR, which never produced an empty prediction yet had 26 zero-overlap cases on LiTS. The paper reports both.

**Liver-constraint analysis is an upper bound.** `liver_constraint.py` uses reference liver masks, which would not be available at deployment. It answers "what would a perfect liver segmentation buy?" — the answer is at most +0.048 Dice.

---

## Citation

```bibtex
@article{kutlu_hcc_crossdataset,
  title  = {Architecture Does Not Determine Cross-Dataset Generalization in
            Hepatocellular Carcinoma Tumor Segmentation},
  author = {Kutlu, H\"{u}seyin and others},
  year   = {2026},
  note   = {under review}
}
```

## License

Code released under the MIT License. The datasets retain their original licences and are not redistributed here.
