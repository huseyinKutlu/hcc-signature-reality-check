import os, glob, argparse
import numpy as np
import SimpleITK as sitk
def metrics(gt_path, pred_path):
    gt_i = sitk.ReadImage(gt_path); pr_i = sitk.ReadImage(pred_path)
    if pr_i.GetSize() != gt_i.GetSize():
        pr_i = sitk.Resample(pr_i, gt_i, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    gt = (sitk.GetArrayFromImage(gt_i) > 0).astype(np.uint8)
    pr = (sitk.GetArrayFromImage(pr_i) > 0).astype(np.uint8)
    inter = np.logical_and(gt, pr).sum(); g, p = gt.sum(), pr.sum()
    dice = 2*inter/(g+p) if (g+p) > 0 else np.nan
    union = np.logical_or(gt, pr).sum(); iou = inter/union if union > 0 else np.nan
    hd95 = assd = np.nan
    if g > 0 and p > 0:
        gt_c = sitk.LabelContour(sitk.Cast(gt_i > 0, sitk.sitkUInt8))
        pr_c = sitk.LabelContour(sitk.Cast(pr_i > 0, sitk.sitkUInt8))
        dt_g = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.Cast(gt_i > 0, sitk.sitkUInt8), squaredDistance=False, useImageSpacing=True))
        dt_p = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.Cast(pr_i > 0, sitk.sitkUInt8), squaredDistance=False, useImageSpacing=True))
        d_p2g = sitk.GetArrayFromImage(dt_g)[sitk.GetArrayFromImage(pr_c) > 0]
        d_g2p = sitk.GetArrayFromImage(dt_p)[sitk.GetArrayFromImage(gt_c) > 0]
        alld = np.concatenate([d_p2g, d_g2p])
        hd95 = float(np.percentile(alld, 95)); assd = float(alld.mean())
    return dict(dice=dice, iou=iou, hd95=hd95, assd=assd, empty_pred=(p == 0),
                vol_gt=float(g), vol_pred=float(p))
def boot(vals, n=2000, seed=42):
    v = np.array([x for x in vals if not np.isnan(x)])
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def main(pred, gt, internal):
    gts = sorted(glob.glob(os.path.join(gt, "*.nii.gz")))
    rows = []
    print(f"{'case':14s} {'dice':>6s} {'hd95mm':>8s} {'assdmm':>8s}")
    for gp in gts:
        case = os.path.basename(gp).replace(".nii.gz", "")
        pp = os.path.join(pred, case + ".nii.gz")
        if not os.path.exists(pp):
            print(f"{case:14s}  [tahmin yok]"); continue
        m = metrics(gp, pp); m["case"] = case; rows.append(m)
        print(f"{case:14s} {m['dice']:6.3f} {m['hd95']:8.2f} {m['assd']:8.2f}" + ("  <-- KACIRILDI" if m['empty_pred'] else ""))
    print(f"\n=== DIS DOGRULAMA (3D-IRCADb, n={len(rows)}) ===")
    for k, unit in [("dice",""),("iou",""),("hd95"," mm"),("assd"," mm")]:
        mean, lo, hi = boot([r[k] for r in rows]); print(f"  {k:5s}: {mean:.3f} [{lo:.3f}, {hi:.3f}]{unit}")
    print(f"  tamamen kacirilan tumor: {sum(r['empty_pred'] for r in rows)}/{len(rows)}")
    ext = boot([r['dice'] for r in rows])[0]
    print(f"\n=== GENELLEME ACIGI ===")
    print(f"  Ic (HCC-TACE-Seg 5-fold CV): {internal:.3f}")
    print(f"  Dis (3D-IRCADb)            : {ext:.3f}")
    print(f"  delta (dis - ic)           : {ext - internal:+.3f}")
    # basit failure-mode ipucu: kucuk tumorlerde dice
    import csv
    with open("ircadb_percase.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["case","dice","hd95","assd","vol_gt_vox","vol_pred_vox","empty_pred"])
        for r in rows: w.writerow([r["case"],round(r["dice"],4),round(r["hd95"],2),round(r["assd"],2),int(r["vol_gt"]),int(r["vol_pred"]),r["empty_pred"]])
    print("per-case CSV yazildi: ircadb_percase.csv")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--internal_dice", type=float, default=0.6848)
    a = ap.parse_args(); main(a.pred, a.gt, a.internal_dice)
