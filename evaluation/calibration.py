import os, glob
import numpy as np, pandas as pd
import SimpleITK as sitk

NBINS, THR = 15, 0.01
EDGES = np.linspace(0, 1, NBINS + 1)

def run(pred_dir, gt_dir, tag):
    bc = np.zeros(NBINS); ba = np.zeros(NBINS); bn = np.zeros(NBINS)
    rows = []
    for npz in sorted(glob.glob(os.path.join(pred_dir, '*.npz'))):
        case = os.path.basename(npz)[:-4]
        gtf = os.path.join(gt_dir, case + '.nii.gz')
        if not os.path.exists(gtf):
            continue
        p = np.load(npz)['probabilities'][1].astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gtf)) > 0
        if p.shape != gt.shape:
            print(f'  [uyari] shape uyusmazligi {case}: {p.shape} vs {gt.shape}')
            continue

        # --- reliability binleri (yalnizca p>THR: trivial arkaplani disla) ---
        m = p > THR
        pv = p[m].astype(np.float64); yv = gt[m].astype(np.float64)
        if pv.size:
            idx = np.clip(np.digitize(pv, EDGES) - 1, 0, NBINS - 1)
            bn += np.bincount(idx, minlength=NBINS)
            bc += np.bincount(idx, weights=pv, minlength=NBINS)
            ba += np.bincount(idx, weights=yv, minlength=NBINS)

        # --- vaka duzeyi ---
        pred = p > 0.5
        inter = np.logical_and(pred, gt).sum()
        dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else np.nan
        rows.append({
            'case': case, 'dice': dice,
            'mean_conf_pred': float(p[pred].mean()) if pred.sum() else np.nan,
            'max_prob': float(p.max()),
            'pred_vol_ml': float(pred.sum()),  # voksel; hacim CSV'de zaten var
            'gt_pos': int(gt.sum())})
        del p, gt, pred

    N = bn.sum()
    conf = np.divide(bc, bn, out=np.zeros(NBINS), where=bn > 0)
    acc  = np.divide(ba, bn, out=np.zeros(NBINS), where=bn > 0)
    ece  = float(np.sum(bn / N * np.abs(acc - conf))) if N else np.nan
    mce  = float(np.max(np.abs(acc - conf)[bn > 0])) if N else np.nan

    if not rows:
        print(f'\n===== {tag} =====\n  [BOS] {pred_dir} icinde islenebilir .npz yok — --save_probabilities ile tekrar tahmin gerekiyor.')
        return None
    df = pd.DataFrame(rows)
    rel = pd.DataFrame({'bin_lo': EDGES[:-1], 'bin_hi': EDGES[1:],
                        'n': bn, 'conf': conf, 'acc': acc})
    rel.to_csv(f'reliability_{tag}.csv', index=False)
    df.to_csv(f'confidence_{tag}.csv', index=False)

    missed = df[df.dice == 0]
    print(f'\n===== {tag}  (n={len(df)}) =====')
    print(f'  ECE = {ece:.4f}   MCE = {mce:.4f}   (p>{THR} voksellerde, {NBINS} bin)')
    print(f'  Ortalama guven (tahmin edilen tumor icinde): {df.mean_conf_pred.mean():.3f}')
    if len(missed):
        print(f'  TAM KACIRILAN {len(missed)} vaka -> ortalama max olasilik: {missed.max_prob.mean():.3f}')
        print(f'     (yuksekse: model yanilirken de kendinden emin)')
    ok = df[df.dice > 0.5]
    if len(ok):
        print(f'  Dice>0.5 vakalar -> ortalama guven: {ok.mean_conf_pred.mean():.3f}')
    if df.mean_conf_pred.notna().sum() > 3:
        from scipy.stats import spearmanr
        r, pp = spearmanr(df.mean_conf_pred, df.dice, nan_policy='omit')
        print(f'  Guven vs Dice: rho={r:.3f}, p={pp:.2e}')
    return ece

for arch, sub in [('resenc', '_prob'), ('umamba', '_umamba'), ('swin', '_swin')]:
    for coh, gt in [('ircadb', 'ircadb_nnunet/labelsTs'), ('lits', 'lits_nnunet/labelsTs')]:
        d = f'{coh}_predictions{sub}'
        if os.path.isdir(d):
            run(d, gt, f'{coh}_{arch}')
        else:
            print(f'[atlandi] {d} yok')
