import os
import numpy as np
import SimpleITK as sitk
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams.update({'font.size':8,'font.family':'sans-serif','savefig.bbox':'tight'})
OUT='figures'; os.makedirs(OUT,exist_ok=True)
WL, WW = 50, 350          # liver window
C = {'resenc':'#2c6fbb','umamba':'#1b9e77','swin':'#d95f02'}
DIRS = {'resenc':'lits_predictions','umamba':'lits_predictions_umamba','swin':'lits_predictions_swin'}
LBL = {'resenc':'ResEnc-M','umamba':'U-Mamba','swin':'SwinUNETR'}

CASES = [
 ('BLITS_032','All three succeed',        {'resenc':0.863,'umamba':0.861,'swin':0.799}, 452.6),
 ('BLITS_006','CNN/SSM succeed, transformer fails', {'resenc':0.862,'umamba':0.853,'swin':0.184}, 17.2),
 ('BLITS_094','Transformer confidently wrong',      {'resenc':0.380,'umamba':0.198,'swin':0.000}, 126.8),
 ('BLITS_020','All three miss a small lesion',      {'resenc':0.000,'umamba':0.000,'swin':0.000}, 0.34),
]

def load(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(p))

def win(x):
    lo, hi = WL-WW/2, WL+WW/2
    return np.clip((x-lo)/(hi-lo), 0, 1)

def contour(mask):
    """thin outline of a binary mask"""
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)

fig, axes = plt.subplots(len(CASES), 4, figsize=(7.2, 1.85*len(CASES)))

for r,(case,caption,dices,vol) in enumerate(CASES):
    ct  = load(f'lits_nnunet/imagesTs/{case}_0000.nii.gz').astype(np.float32)
    gt  = load(f'lits_nnunet/labelsTs/{case}.nii.gz') > 0
    preds = {a: load(os.path.join(d, case+'.nii.gz')) > 0 for a,d in DIRS.items()}

    # slice with the largest reference tumor area
    areas = gt.sum(axis=(1,2))
    z = int(np.argmax(areas)) if areas.max() > 0 else ct.shape[0]//2
    base, g = win(ct[z]), gt[z]

    ax = axes[r,0]
    ax.imshow(base, cmap='gray', vmin=0, vmax=1)
    if g.any():
        ov = np.zeros((*g.shape,4)); ov[contour(g)] = [1,1,0,1]
        ax.imshow(ov)
    ax.set_ylabel(f'{case}\n{caption}\n({vol:.1f} mL)', fontsize=6.5)
    ax.set_title('CT + reference', fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])

    for c,a in enumerate(['resenc','umamba','swin'], start=1):
        ax = axes[r,c]
        ax.imshow(base, cmap='gray', vmin=0, vmax=1)
        p = preds[a][z]
        if p.any():
            rgb = tuple(int(C[a][i:i+2],16)/255 for i in (1,3,5))
            ov = np.zeros((*p.shape,4)); ov[p] = [*rgb, 0.55]
            ax.imshow(ov)
        if g.any():
            ov2 = np.zeros((*g.shape,4)); ov2[contour(g)] = [1,1,0,1]
            ax.imshow(ov2)
        if not p.any():
            ax.text(.5,.06,'no prediction on this slice', transform=ax.transAxes,
                    ha='center', fontsize=5.5, color='w')
        elif case=='BLITS_020' and a=='swin':
            ax.text(.5,.06,'false positive in normal parenchyma', transform=ax.transAxes,
                    ha='center', fontsize=5.5, color='#ffcc66')
        elif case=='BLITS_094' and a=='swin':
            ax.text(.5,.06,'zero overlap; 6.5x reference volume', transform=ax.transAxes,
                    ha='center', fontsize=5.5, color='#ffcc66')
        ax.set_title(f'{LBL[a]}  DSC {dices[a]:.3f}', fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])

    # crop all panels of the row around the reference lesion
    if g.any():
        ys,xs = np.where(g)
        if case == 'BLITS_094':          # tahmin referanstan cok uzakta: ikisini de kadraja al
            ay = np.concatenate([ys] + [np.where(preds[a][z])[0] for a in preds if preds[a][z].any()])
            ax_ = np.concatenate([xs] + [np.where(preds[a][z])[1] for a in preds if preds[a][z].any()])
            cy, cx = (ay.min()+ay.max())//2, (ax_.min()+ax_.max())//2
            half = int(0.62*max(ay.max()-ay.min(), ax_.max()-ax_.min()))
        else:
            cy,cx = (ys.min()+ys.max())//2, (xs.min()+xs.max())//2
            half = max(70, int(1.9*max(ys.max()-ys.min(), xs.max()-xs.min())))
        for c in range(4):
            axes[r,c].set_xlim(max(0,cx-half), min(base.shape[1],cx+half))
            axes[r,c].set_ylim(min(base.shape[0],cy+half), max(0,cy-half))

handles=[Patch(facecolor='none',edgecolor='yellow',label='Reference (ground truth)')]+\
        [Patch(facecolor=C[a],alpha=.55,label=LBL[a]) for a in ['resenc','umamba','swin']]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
           fontsize=7, bbox_to_anchor=(.5,-.02))
fig.tight_layout()
fig.savefig(f'{OUT}/Fig6_qualitative.png', dpi=300)
fig.savefig(f'{OUT}/Fig6_qualitative.pdf')
print('-> figures/Fig6_qualitative.png / .pdf')
