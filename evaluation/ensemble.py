"""
Three-architecture probability ensemble.

Averages the softmax tumour-probability maps of the convolutional, state-space
and transformer models and evaluates the result, to test whether combining
architectural families recovers any cross-dataset performance.

Also reports pairwise ensembles, so a gain (or its absence) can be attributed.

Usage:
    python ensemble.py                 # both cohorts
    python ensemble.py --cohort lits   # one cohort
"""
import os
import glob
import argparse
import itertools

import numpy as np
import pandas as pd
import SimpleITK as sitk

# probability directories written with --save_probabilities
PROB = {'resenc': '{coh}_predictions_prob',
        'umamba': '{coh}_predictions_umamba',
        'swin':   '{coh}_predictions_swin'}
ARCH = list(PROB)


def dice(p, g):
    s = p.sum() + g.sum()
    return np.nan if s == 0 else 2.0 * np.logical_and(p, g).sum() / s


def load_prob(path):
    """Return the foreground (tumour) probability map from an nnU-Net .npz."""
    z = np.load(path)
    key = 'probabilities' if 'probabilities' in z else list(z.keys())[0]
    return z[key][1].astype(np.float32)


def run(coh, thr):
    gt_files = sorted(glob.glob(f'{coh}_nnunet/labelsTs/*.nii.gz'))
    combos = [(a,) for a in ARCH] + \
             [c for c in itertools.combinations(ARCH, 2)] + \
             [tuple(ARCH)]
    rows = []
    skipped = 0

    for gtf in gt_files:
        case = os.path.basename(gtf)[:-7]
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gtf)) > 0

        probs = {}
        ok = True
        for a in ARCH:
            f = os.path.join(PROB[a].format(coh=coh), case + '.npz')
            if not os.path.exists(f):
                ok = False
                break
            p = load_prob(f)
            if p.shape != gt.shape:
                ok = False
                break
            probs[a] = p
        if not ok:
            skipped += 1
            continue

        r = {'case': case}
        for combo in combos:
            mean_p = np.mean([probs[a] for a in combo], axis=0)
            r['+'.join(combo)] = dice(mean_p > thr, gt)
        rows.append(r)
        del probs

    df = pd.DataFrame(rows)
    if df.empty:
        print(f'[{coh}] degerlendirilebilir vaka yok (atlanan: {skipped})')
        return None

    print(f'\n===== {coh.upper()} (n={len(df)}, atlanan={skipped}, esik={thr}) =====')
    single = {a: df[a].mean() for a in ARCH}
    best_single = max(single, key=single.get)
    for combo in combos:
        k = '+'.join(combo)
        m, nfail = df[k].mean(), (df[k] == 0).sum()
        tag = ''
        if len(combo) > 1:
            d = m - single[best_single]
            tag = f'   (en iyi tekile gore {d:+.3f})'
        print(f'  {k:26s} Dice {m:.3f}  tam kacirma {nfail:3d}{tag}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', choices=['ircadb', 'lits', 'both'], default='both')
    ap.add_argument('--thr', type=float, default=0.5)
    a = ap.parse_args()

    cohs = ['ircadb', 'lits'] if a.cohort == 'both' else [a.cohort]
    out = []
    for coh in cohs:
        df = run(coh, a.thr)
        if df is not None:
            df.insert(0, 'cohort', coh)
            out.append(df)
    if out:
        pd.concat(out, ignore_index=True).to_csv('ensemble_percase.csv', index=False)
        print('\n-> ensemble_percase.csv yazildi')
