"""
Failure-mode analizi — genelleme açığı tümör boyutuyla nasıl değişiyor?
======================================================================
per-case CSV'ler (dice + vol_gt_vox) + mapping CSV'ler (mL) birleştirilir.
Çıktı: boyut gruplarına göre Dice + kaçırma oranı, Spearman korelasyonu.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

def load(percase, mapping, mlcol, idcol, kohort):
    pc = pd.read_csv(percase)
    mp = pd.read_csv(mapping)
    mp = mp.rename(columns={mlcol: "tumor_ml", idcol: "case"})
    df = pc.merge(mp[["case", "tumor_ml"]], on="case", how="left")
    df["kohort"] = kohort
    return df[["case", "kohort", "dice", "tumor_ml", "empty_pred"]]

base = "/home/hkutlu/Desktop/JILTI/"
# yollar kullanıcıda; burada sadece iskelet — gerçek çalıştırma kullanıcı makinesinde
ircadb = load(base+"ircadb_pc.csv", base+"ircadb_nnunet/ircadb_mapping.csv", "tumor_ml", "case", "IRCADb")
lits   = load(base+"lits_pc.csv",   base+"lits_nnunet/lits_mapping.csv",     "tumor_ml", "case", "LiTS")
df = pd.concat([ircadb, lits], ignore_index=True)
df = df.dropna(subset=["tumor_ml"])

print(f"Toplam dış vaka: {len(df)}  (IRCADb {len(ircadb)}, LiTS {len(lits)})\n")

# boyut grupları (klinik olarak anlamlı eşikler)
bins   = [0, 5, 20, 50, 1e9]
labels = ["<5 mL (cok kucuk)", "5-20 mL (kucuk)", "20-50 mL (orta)", ">50 mL (buyuk)"]
df["grup"] = pd.cut(df["tumor_ml"], bins=bins, labels=labels)

print("=== BOYUT GRUBUNA GORE (iki kohort birlesik) ===")
print(f"{'grup':22s} {'n':>4s} {'ort.Dice':>9s} {'medyanDice':>11s} {'kacirma%':>9s}")
for g in labels:
    sub = df[df["grup"] == g]
    if len(sub) == 0: continue
    miss = 100*(sub["dice"] == 0).mean()
    print(f"{g:22s} {len(sub):>4d} {sub['dice'].mean():>9.3f} {sub['dice'].median():>11.3f} {miss:>8.0f}%")

rho, p = spearmanr(df["tumor_ml"], df["dice"])
print(f"\nSpearman(tumor_ml, Dice): rho={rho:.3f}, p={p:.2e}")
print("-> pozitif ve anlamli ise: buyuk tumor = yuksek Dice; kucukler cokuyor")

# kohort bazinda ortalama
print("\n=== KOHORT BAZINDA ===")
for k in ["IRCADb", "LiTS"]:
    sub = df[df["kohort"] == k]
    print(f"  {k}: n={len(sub)}, ort.Dice={sub['dice'].mean():.3f}, kacirma={100*(sub['dice']==0).mean():.0f}%")

df.to_csv(base+"failure_mode_merged.csv", index=False)
print("\nBirlesik CSV: failure_mode_merged.csv")
