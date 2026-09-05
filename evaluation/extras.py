import numpy as np, pandas as pd, glob, os
import SimpleITK as sitk
from scipy.stats import spearmanr

ARCH=['resenc','umamba','swin']
sel=pd.read_csv('selective_input.csv'); sel['mean_conf_pred']=sel.mean_conf_pred.fillna(0)

# ============ 1) Tablo 7 eksik hucreleri ============
print("="*60); print("1) RISK-COVERAGE (LiTS) - tum mimariler")
for a in ARCH:
    d=sel[(sel.cohort=='lits')&(sel.arch==a)].sort_values('mean_conf_pred',ascending=False).reset_index(drop=True)
    row=[]
    for c in [.5,.4,.3]:
        k=int(round(len(d)*c))
        row.append(f"cov{int(c*100)}: {d.dice[:k].mean():.3f} ({(d.dice[:k]==0).mean()*100:.1f}%)")
    print(f"  {a:8s} "+"  ".join(row))

# ============ 2) Esik transferi: IRCADb'de sec, LiTS'te uygula ============
print("\n"+"="*60); print("2) ESIK TRANSFERI (IRCADb -> LiTS)")
for a in ARCH:
    I=sel[(sel.cohort=='ircadb')&(sel.arch==a)]
    L=sel[(sel.cohort=='lits')&(sel.arch==a)]
    for target_cov in [.7,.5]:
        thr=np.quantile(I.mean_conf_pred, 1-target_cov)   # IRCADb'de %X kapsam veren esik
        keep=L[L.mean_conf_pred>=thr]
        if len(keep)==0: print(f"  {a:8s} hedef{int(target_cov*100)}: bos"); continue
        print(f"  {a:8s} hedef kapsam {int(target_cov*100)}% -> esik={thr:.3f} | "
              f"LiTS'te gerceklesen kapsam {100*len(keep)/len(L):4.1f}%  "
              f"dice={keep.dice.mean():.3f} (tum kohort {L.dice.mean():.3f})  "
              f"fail={100*(keep.dice==0).mean():.1f}%")

# ============ 3) Alan kaymasi: kohort karakteristikleri ============
print("\n"+"="*60); print("3) KOHORT KARAKTERISTIKLERI (alan kaymasi)")
def cohort_stats(img_dir, lab_dir, name, nmax=None):
    imgs=sorted(glob.glob(os.path.join(img_dir,'*.nii.gz')))
    if nmax: imgs=imgs[:nmax]
    sp_z,sp_xy,hu_m,hu_s,vols,nles=[],[],[],[],[],[]
    for f in imgs:
        base=os.path.basename(f).replace('_0000.nii.gz','.nii.gz')
        lf=os.path.join(lab_dir,base)
        if not os.path.exists(lf): continue
        im=sitk.ReadImage(f); sp=im.GetSpacing()
        arr=sitk.GetArrayFromImage(im).astype(np.float32)
        lab=sitk.GetArrayFromImage(sitk.ReadImage(lf))>0
        if lab.sum()==0: continue
        sp_z.append(sp[2]); sp_xy.append(sp[0])
        hu_m.append(float(arr[lab].mean())); hu_s.append(float(arr[lab].std()))
        vols.append(lab.sum()*sp[0]*sp[1]*sp[2]/1000.0)
        cc=sitk.ConnectedComponent(sitk.ReadImage(lf)>0)
        nles.append(int(sitk.GetArrayFromImage(cc).max()))
        del arr,lab
    def q(x): return f"{np.median(x):.1f} [{np.percentile(x,25):.1f}-{np.percentile(x,75):.1f}]"
    print(f"\n  --- {name} (n={len(vols)}) ---")
    print(f"    slice thickness (mm) : {q(sp_z)}")
    print(f"    in-plane spacing (mm): {np.median(sp_xy):.3f}")
    print(f"    tumor HU mean        : {q(hu_m)}")
    print(f"    tumor HU sd (within) : {q(hu_s)}")
    print(f"    lesion volume (mL)   : {q(vols)}")
    print(f"    lesions per case     : {q(nles)}")
    print(f"    <5 mL orani          : {100*np.mean(np.array(vols)<5):.1f}%")

cohort_stats('nnUNet_raw/Dataset501_HCCtumor/imagesTr','nnUNet_raw/Dataset501_HCCtumor/labelsTr','HCC-TACE-Seg (development)')
cohort_stats('ircadb_nnunet/imagesTs','ircadb_nnunet/labelsTs','3D-IRCADb')
cohort_stats('lits_nnunet/imagesTs','lits_nnunet/labelsTs','LiTS')
