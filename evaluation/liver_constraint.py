"""
Anatomical (liver-mask) constraint analysis.

Restricts each model's tumour predictions to the reference liver mask and
recomputes Dice, complete-failure counts, and the fraction of predicted
tumour volume that fell outside the liver.

Note: the liver mask is taken from the reference annotation, so this is an
upper-bound ("perfect liver segmentation") analysis, not a deployable
pipeline. Report it as such.

Usage:
    python liver_constraint.py --lits_root ~/Desktop/JILTI/Task03_Liver
    python liver_constraint.py --lits_root ... --margin 3      # dilate liver by 3 voxels
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_dilation

ARCH = {'resenc': 'predictions', 'umamba': 'predictions_umamba', 'swin': 'predictions_swin'}


def read_dicom_series(dir_path):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(dir_path)
    if not ids:
        r.SetFileNames(sorted(glob.glob(os.path.join(dir_path, "*"))))
    else:
        r.SetFileNames(r.GetGDCMSeriesFileNames(dir_path, ids[0]))
    return r.Execute()


def find_mapping(cohort):
    """Locate the mapping CSV written by the conversion scripts."""
    for pat in (f'{cohort}_nnunet/{cohort}_mapping.csv',
                f'{cohort}_nnunet/*mapping*.csv',
                f'{cohort}_mapping.csv'):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def build_ircadb_livers(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mp = find_mapping('ircadb')
    if mp is None:
        print('  [ATLA] ircadb mapping CSV bulunamadi')
        return
    m = pd.read_csv(mp)
    idcol = 'ircad_id' if 'ircad_id' in m.columns else m.columns[1]
    n = 0
    for _, row in m.iterrows():
        case, pid = row['case'], str(row[idcol])
        pdir = os.path.join(root, pid if pid.startswith('3Dircadb1.') else f'3Dircadb1.{pid}')
        liver_dir = os.path.join(pdir, 'MASKS_DICOM', 'liver')
        if not os.path.isdir(liver_dir):
            print(f'  [YOK] {case}: {liver_dir}')
            continue
        ref = sitk.ReadImage(f'ircadb_nnunet/labelsTs/{case}.nii.gz')
        liv = sitk.GetArrayFromImage(read_dicom_series(liver_dir)) > 0
        if liv.shape != sitk.GetArrayFromImage(ref).shape:
            print(f'  [SEKIL UYUSMAZ] {case}: {liv.shape} vs {sitk.GetArrayFromImage(ref).shape}')
            continue
        im = sitk.GetImageFromArray(liv.astype(np.uint8))
        im.CopyInformation(ref)
        sitk.WriteImage(im, os.path.join(out_dir, f'{case}.nii.gz'))
        n += 1
    print(f'  IRCADb karaciger maskesi: {n}')


def build_lits_livers(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mp = find_mapping('lits')
    if mp is None:
        print('  [ATLA] lits mapping CSV bulunamadi')
        return
    m = pd.read_csv(mp)
    srccol = 'lits_file' if 'lits_file' in m.columns else m.columns[1]
    n = 0
    for _, row in m.iterrows():
        case, src = row['case'], row[srccol]
        lp = os.path.join(root, 'labelsTr', src)
        if not os.path.exists(lp):
            print(f'  [YOK] {case}: {lp}')
            continue
        lab = sitk.ReadImage(lp)
        liv = sitk.GetArrayFromImage(lab) >= 1          # liver (1) + tumour (2)
        im = sitk.GetImageFromArray(liv.astype(np.uint8))
        im.CopyInformation(lab)
        sitk.WriteImage(im, os.path.join(out_dir, f'{case}.nii.gz'))
        n += 1
    print(f'  LiTS karaciger maskesi: {n}')


def dice(p, g):
    s = p.sum() + g.sum()
    return np.nan if s == 0 else 2.0 * np.logical_and(p, g).sum() / s


def evaluate(coh, liver_dir, margin):
    rows = []
    for gtf in sorted(glob.glob(f'{coh}_nnunet/labelsTs/*.nii.gz')):
        case = os.path.basename(gtf)[:-7]
        lf = os.path.join(liver_dir, f'{case}.nii.gz')
        if not os.path.exists(lf):
            continue
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gtf)) > 0
        liver = sitk.GetArrayFromImage(sitk.ReadImage(lf)) > 0
        if margin:
            liver = binary_dilation(liver, iterations=margin)
        r = {'case': case, 'cohort': coh}
        for a, sub in ARCH.items():
            pf = os.path.join(f'{coh}_{sub}', f'{case}.nii.gz')
            if not os.path.exists(pf):
                continue
            p = sitk.GetArrayFromImage(sitk.ReadImage(pf)) > 0
            if p.shape != gt.shape:
                continue
            pc = np.logical_and(p, liver)
            r[f'{a}_raw'] = dice(p, gt)
            r[f'{a}_liv'] = dice(pc, gt)
            r[f'{a}_out'] = 0.0 if p.sum() == 0 else 1 - pc.sum() / p.sum()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lits_root', required=True,
                    help='MSD Task03 root (same path given to convert_lits.py --root)')
    ap.add_argument('--ircadb_root', default='3Dircadb1')
    ap.add_argument('--margin', type=int, default=0,
                    help='dilate the liver mask by N voxels before constraining')
    a = ap.parse_args()

    print('Karaciger maskeleri olusturuluyor...')
    build_ircadb_livers(a.ircadb_root, 'ircadb_liver')
    build_lits_livers(os.path.expanduser(a.lits_root), 'lits_liver')

    out = []
    for coh, ld in [('ircadb', 'ircadb_liver'), ('lits', 'lits_liver')]:
        df = evaluate(coh, ld, a.margin)
        if df.empty:
            print(f'\n[{coh}] degerlendirilebilir vaka yok')
            continue
        out.append(df)
        print(f'\n===== {coh.upper()} (n={len(df)}, margin={a.margin}) =====')
        for arch in ARCH:
            if f'{arch}_raw' not in df.columns:
                continue
            raw, liv, of = df[f'{arch}_raw'], df[f'{arch}_liv'], df[f'{arch}_out']
            print(f'  {arch:8s} Dice {raw.mean():.3f} -> {liv.mean():.3f} '
                  f'({liv.mean() - raw.mean():+.3f}) | '
                  f'tam kacirma {(raw == 0).sum()} -> {(liv == 0).sum()} | '
                  f'karaciger disi tahmin {of.mean() * 100:.1f}% '
                  f'(medyan {of.median() * 100:.1f}%)')

    if out:
        pd.concat(out, ignore_index=True).to_csv('liver_constrained.csv', index=False)
        print('\n-> liver_constrained.csv yazildi')


if __name__ == '__main__':
    main()
