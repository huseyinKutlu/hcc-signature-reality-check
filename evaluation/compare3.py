import glob, os, itertools
import numpy as np, pandas as pd
import SimpleITK as sitk
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare

ARCH = ['resenc', 'umamba', 'swin']

def load(coh, arch):
    d = pd.read_csv(f'percase_{coh}_{arch}.csv')
    d.columns = [c.strip().lower() for c in d.columns]
    return d[['case', 'dice']].rename(columns={'dice': arch})

def gt_volume_ml(gt_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(gt_dir, '*.nii.gz'))):
        img = sitk.ReadImage(f); a = sitk.GetArrayFromImage(img) > 0
        sp = img.GetSpacing()
        rows.append({'case': os.path.basename(f)[:-7],
                     'vol_ml': a.sum() * sp[0]*sp[1]*sp[2] / 1000.0})
    return pd.DataFrame(rows)

allm = []
for coh, gtd in [('ircadb','ircadb_nnunet/labelsTs'), ('lits','lits_nnunet/labelsTs')]:
    m = load(coh, ARCH[0])
    for a in ARCH[1:]:
        m = m.merge(load(coh, a), on='case')
    m = m.merge(gt_volume_ml(gtd), on='case', how='left')
    m['cohort'] = coh; allm.append(m)

    print(f"\n===== {coh.upper()} (n={len(m)}) =====")
    for a in ARCH:
        print(f"  {a:8s} ort={m[a].mean():.3f}  tam kacirma={(m[a]==0).sum()}")
    print(f"  Friedman: p={friedmanchisquare(*[m[a] for a in ARCH]).pvalue:.4f}")
    print("  Vaka bazli Spearman (mimari ciftleri):")
    for a,b in itertools.combinations(ARCH,2):
        r,p = spearmanr(m[a], m[b]); print(f"    {a:8s} vs {b:8s}: rho={r:.3f} (p={p:.1e})")
    print("  Eslesmis Wilcoxon:")
    for a,b in itertools.combinations(ARCH,2):
        print(f"    {a:8s} vs {b:8s}: p={wilcoxon(m[a], m[b]).pvalue:.4f}")
    print("  Hacim vs Dice:")
    for a in ARCH:
        r,p = spearmanr(m.vol_ml, m[a]); print(f"    {a:8s}: rho={r:.3f} (p={p:.1e})")

df = pd.concat(allm, ignore_index=True)
df['vol_grp'] = pd.cut(df.vol_ml, [0,5,20,50,np.inf], labels=['<5 mL','5-20 mL','20-50 mL','>50 mL'])
print(f"\n===== HACIM KATMANLARI (n={len(df)}) =====")
agg = {'n': ('case','size')}
for a in ARCH:
    agg[f'dice_{a}'] = (a,'mean'); agg[f'miss_{a}'] = (a, lambda s: (s==0).mean())
print(df.groupby('vol_grp', observed=True).agg(**agg).round(3).to_string())
df.to_csv('percase_combined3.csv', index=False)
print("\n-> percase_combined3.csv yazildi")
