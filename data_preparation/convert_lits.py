import os, glob, argparse, csv
import numpy as np
import SimpleITK as sitk
def main(root, out):
    img_dir = os.path.join(out, "imagesTs"); lab_dir = os.path.join(out, "labelsTs")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(root, "imagesTr", "*.nii.gz")))
    imgs = [f for f in imgs if not os.path.basename(f).startswith("._")]
    print("Bulunan etiketli vaka:", len(imgs))
    mapping, n_ok, n_skip = [], 0, 0
    for ip in imgs:
        base = os.path.basename(ip)
        lp = os.path.join(root, "labelsTr", base)
        if not os.path.exists(lp):
            print(f"  [ATLA] {base}: label yok"); n_skip += 1; continue
        lab_img = sitk.ReadImage(lp); lab = sitk.GetArrayFromImage(lab_img)
        tumor = (lab == 2).astype(np.uint8)
        if tumor.sum() == 0:
            print(f"  [DISLA] {base}: tumor yok"); n_skip += 1; continue
        ct = sitk.ReadImage(ip); sp = ct.GetSpacing(); vox_ml = sp[0]*sp[1]*sp[2]/1000.0
        vol = tumor.sum()*vox_ml; case = f"BLITS_{n_ok+1:03d}"
        tim = sitk.GetImageFromArray(tumor); tim.CopyInformation(lab_img)
        sitk.WriteImage(ct,  os.path.join(img_dir, f"{case}_0000.nii.gz"))
        sitk.WriteImage(tim, os.path.join(lab_dir, f"{case}.nii.gz"))
        mapping.append((case, base, round(sp[2],2), round(vol,1)))
        print(f"  {case} ({base}): z={round(sp[2],2)}, tumor={vol:.1f} mL")
        n_ok += 1
    with open(os.path.join(out, "lits_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case","lits_file","sz","tumor_ml"]); w.writerows(mapping)
    print(f"\n=== OZET === dis dogrulama vakasi: {n_ok} | dislanan: {n_skip}")
    if mapping:
        vols = sorted(m[3] for m in mapping)
        print(f"tumor hacmi mL — min {vols[0]}, medyan {vols[len(vols)//2]}, max {vols[-1]}")
    print("Cikti:", out)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(a.root, a.out)
