import os, glob, argparse, csv
import numpy as np
import SimpleITK as sitk

def read_dicom_series(dir_path):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(dir_path)
    if not ids:
        files = sorted(glob.glob(os.path.join(dir_path, "*")))
        r.SetFileNames(files)
    else:
        r.SetFileNames(r.GetGDCMSeriesFileNames(dir_path, ids[0]))
    return r.Execute()

def load_mask_union(masks_root, ref_img):
    tumor_dirs = sorted(glob.glob(os.path.join(masks_root, "livertumor*")))
    if not tumor_dirs:
        return None, []
    union = None
    for td in tumor_dirs:
        m = read_dicom_series(td)
        m = sitk.Resample(m, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        arr = (sitk.GetArrayFromImage(m) > 0).astype(np.uint8)
        union = arr if union is None else np.logical_or(union, arr).astype(np.uint8)
    return union, [os.path.basename(t) for t in tumor_dirs]

def main(root, out):
    img_dir = os.path.join(out, "imagesTs"); lab_dir = os.path.join(out, "labelsTs")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    patients = sorted(glob.glob(os.path.join(root, "3Dircadb1.*")),
                      key=lambda p: int(p.split(".")[-1]))
    print("Bulunan hasta klasörü:", len(patients))
    mapping, n_ok, n_skip = [], 0, 0
    for p in patients:
        name = os.path.basename(p)
        ct_dir = os.path.join(p, "PATIENT_DICOM"); masks = os.path.join(p, "MASKS_DICOM")
        if not os.path.isdir(ct_dir) or not os.path.isdir(masks):
            print(f"  [ATLA] {name}: PATIENT_DICOM/MASKS_DICOM yok"); n_skip += 1; continue
        ct = read_dicom_series(ct_dir)
        tumor, used = load_mask_union(masks, ct)
        if tumor is None:
            print(f"  [DISLA] {name}: livertumor yok (tumorsuz)"); n_skip += 1; continue
        sp = ct.GetSpacing(); vox_ml = sp[0]*sp[1]*sp[2]/1000.0
        vol = tumor.sum()*vox_ml
        case = f"BIRCAD_{n_ok+1:03d}"
        lab = sitk.GetImageFromArray(tumor.astype(np.uint8)); lab.CopyInformation(ct)
        sitk.WriteImage(ct,  os.path.join(img_dir, f"{case}_0000.nii.gz"))
        sitk.WriteImage(lab, os.path.join(lab_dir, f"{case}.nii.gz"))
        mapping.append((case, name, len(used), round(sp[2],2), round(vol,1)))
        print(f"  {case} ({name}): {len(used)} tumor klasoru, z={round(sp[2],2)}, tumor={vol:.1f} mL")
        n_ok += 1
    with open(os.path.join(out, "ircadb_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case","ircad_id","n_tumor_dirs","sz","tumor_ml"]); w.writerows(mapping)
    print(f"\n=== OZET === dis dogrulama vakasi: {n_ok} | dislanan: {n_skip}")
    if mapping:
        vols = sorted(m[4] for m in mapping)
        print(f"tumor hacmi mL — min {vols[0]}, medyan {vols[len(vols)//2]}, max {vols[-1]}")
    print("Cikti:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(a.root, a.out)
