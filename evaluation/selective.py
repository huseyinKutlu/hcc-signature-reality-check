import numpy as np, pandas as pd
from scipy.stats import spearmanr

ARCH = ['resenc', 'umamba', 'swin']
COH  = {'ircadb': 'ircadb', 'lits': 'lits'}

def load(coh, arch):
    d = pd.read_csv(f'percase_{coh}_{arch}.csv'); d.columns=[c.strip().lower() for c in d.columns]
    c = pd.read_csv(f'confidence_{coh}_{arch}.csv')
    m = d[['case','dice']].merge(c[['case','mean_conf_pred','max_prob','pred_vol_ml']], on='case')
    m['arch']=arch; m['cohort']=coh
    return m

df = pd.concat([load(c,a) for c in COH for a in ARCH], ignore_index=True)
df['mean_conf_pred'] = df.mean_conf_pred.fillna(0.0)   # bos tahmin -> guven 0

def risk_coverage(sub, score_col, higher_is_better=True):
    s = sub.sort_values(score_col, ascending=not higher_is_better).reset_index(drop=True)
    out=[]
    for cov in [1.0,0.9,0.8,0.7,0.6,0.5]:
        k = max(1,int(round(len(s)*cov)))
        out.append({'coverage':cov, 'n':k, 'dice':s.dice[:k].mean(),
                    'miss_rate':(s.dice[:k]==0).mean()})
    return pd.DataFrame(out)

for coh in COH:
    print(f"\n########## {coh.upper()} ##########")
    for a in ARCH:
        sub = df[(df.cohort==coh)&(df.arch==a)]
        r_conf,_ = spearmanr(sub.mean_conf_pred, sub.dice)
        r_vol ,_ = spearmanr(sub.pred_vol_ml,   sub.dice)
        print(f"\n--- {a} (n={len(sub)}) ---")
        print(f"  secim skoru olarak guven : rho={r_conf:.3f}")
        print(f"  secim skoru olarak tahmini hacim: rho={r_vol:.3f}")
        print("  [GUVEN ile secim]"); print(risk_coverage(sub,'mean_conf_pred').round(3).to_string(index=False))
        print("  [TAHMINI HACIM ile secim]"); print(risk_coverage(sub,'pred_vol_ml').round(3).to_string(index=False))

df.to_csv('selective_input.csv', index=False)
print("\n-> selective_input.csv yazildi")
