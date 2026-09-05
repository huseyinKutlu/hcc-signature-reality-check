"""
Adım 3+4 (highdicom sürümü) — geometri-güvenli SEG->CT hizalama + nnU-Net + envanter
====================================================================================
Neden değişti: pydicom_seg bazı SEG'lerin z-eksenini TERS inşa ediyordu -> maske CT
dışına düşüp 0 mL veriyordu. highdicom, her SEG frame'ini referans CT diliminin
SOPInstanceUID'i üzerinden yerleştirir -> geometri tahmini YOK, flip/offset YOK.

BONUS: pydicom_seg gerekmez. Bu ortamda:
    pip install highdicom SimpleITK pydicom   (numpy<2 / pydicom<3 kısıtı ARTIK YOK)

Kullanım:
    python prepare_nnunet_dataset.py --limit 3
    python prepare_nnunet_dataset.py
"""
import os, glob, json, argparse, csv
import numpy as np
import SimpleITK as sitk
import pydicom
import highdicom as hd

sitk.ProcessObject.SetGlobalWarningDisplay(False)

DEST = "/home/hkutlu/Desktop/JILTI/HCC-TACE-Seg"
TUMOR_LABEL = "Mass"
OUT_ROOT = os.environ.get("nnUNet_raw", os.path.join(os.getcwd(), "nnUNet_raw"))
DATASET = "Dataset501_HCCtumor"


def index_series(root):
    idx = {}
    for d in {os.path.dirname(f) for f in glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True)}:
        fs = glob.glob(os.path.join(d, "*.dcm"))
        if not fs:
            continue
        try:
            h = pydicom.dcmread(fs[0], stop_before_pixels=True)
            idx[h.SeriesInstanceUID] = (d, getattr(h, "Modality", ""))
        except Exception:
            pass
    return idx


def mass_segment_number(seg):
    for s in seg.SegmentSequence:
        if str(getattr(s, "SegmentLabel", "")) == TUMOR_LABEL:
            return int(s.SegmentNumber)
    return None


def build_case(seg_path, ct_dir):
    """SOP-tabanlı hizalama. Klasördeki her GDCM alt-serisini dener, en dolu maskeyi seçer."""
    seg = hd.seg.segread(seg_path)
    mass = mass_segment_number(seg)
    if mass is None:
        return None
    r = sitk.ImageSeriesReader()
    best = None  # (voxels, ct_img, mask_np)
    for sid in r.GetGDCMSeriesIDs(ct_dir):
        files = r.GetGDCMSeriesFileNames(ct_dir, sid)
        sops = [pydicom.dcmread(f, stop_before_pixels=True).SOPInstanceUID for f in files]
        try:
            mask = seg.get_pixels_by_source_instance(
                source_sop_instance_uids=sops,
                segment_numbers=[mass],
                combine_segments=True,
                assert_missing_frames_are_empty=True,
                ignore_spatial_locations=True,
                skip_overlap_checks=True,
            )
        except Exception:
            continue
        mb = (mask > 0).astype(np.uint8)
        v = int(mb.sum())
        if best is None or v > best[0]:
            rr = sitk.ImageSeriesReader(); rr.SetFileNames(files)
            best = (v, rr.Execute(), mb)
    return best


def main(limit):
    idx = index_series(DEST)
    seg_files = [f for f in glob.glob(os.path.join(DEST, "**", "*.dcm"), recursive=True)
                 if idx.get(pydicom.dcmread(f, stop_before_pixels=True).SeriesInstanceUID, ("", ""))[1] == "SEG"]
    print("SEG dosyası:", len(seg_files))

    img_dir = os.path.join(OUT_ROOT, DATASET, "imagesTr")
    lab_dir = os.path.join(OUT_ROOT, DATASET, "labelsTr")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)

    mapping, n_ok, still_empty = [], 0, []
    for seg in sorted(seg_files):
        if limit and n_ok >= limit:
            break
        d = pydicom.dcmread(seg, stop_before_pixels=True)
        pid = getattr(d, "PatientID", "?")
        try:
            ref = d.ReferencedSeriesSequence[0].SeriesInstanceUID
        except Exception:
            print(f"  [ATLA] {pid}: referans CT yok"); continue
        if ref not in idx:
            print(f"  [ATLA] {pid}: referans CT indirilmemiş"); continue

        res = build_case(seg, idx[ref][0])
        if res is None:
            print(f"  [ATLA] {pid}: '{TUMOR_LABEL}' yok / okunamadı"); continue
        _, ct, mask_np = res

        # highdicom (n,rows,cols) == sitk (z,y,x); CT geometrisini kopyala
        if mask_np.shape != tuple(ct.GetSize()[::-1]):
            print(f"  [UYARI] {pid}: maske {mask_np.shape} vs CT {ct.GetSize()[::-1]} boyut uyuşmuyor")
        sp = ct.GetSpacing(); vox_ml = sp[0]*sp[1]*sp[2]/1000.0
        vol = mask_np.sum()*vox_ml

        # GÜVENLİK: boş maske üreten vakayı ASLA yazma (bozuk etiketle eğitim olmaz)
        if mask_np.sum() == 0:
            still_empty.append(pid)
            print(f"  [DIŞLA] {pid}: maske boş (kaynak seri uyuşmazlığı) -> yazılmadı")
            continue

        case = f"HCC_{n_ok+1:03d}"
        lab = sitk.GetImageFromArray(mask_np); lab.CopyInformation(ct)
        sitk.WriteImage(ct,  os.path.join(img_dir, f"{case}_0000.nii.gz"))
        sitk.WriteImage(lab, os.path.join(lab_dir, f"{case}.nii.gz"))
        mapping.append((case, pid, round(sp[0],2), round(sp[1],2), round(sp[2],2), int(mask_np.sum()), round(vol,1)))
        print(f"  {case} ({pid}): spacing={tuple(round(x,2) for x in sp)} tümör={vol:.1f} mL")
        n_ok += 1

    with open(os.path.join(OUT_ROOT, DATASET, "dataset.json"), "w") as f:
        json.dump({"channel_names": {"0": "CT"},
                   "labels": {"background": 0, "tumor": 1},
                   "numTraining": n_ok, "file_ending": ".nii.gz"}, f, indent=2)
    with open("hcc_case_mapping.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case","patient_id","sx","sy","sz","tumor_voxels","tumor_ml"])
        w.writerows(mapping)

    print(f"\n=== ÖZET ===  başarılı vaka: {n_ok}")
    if mapping:
        vols = sorted(m[6] for m in mapping)
        print(f"tümör hacmi mL — min {vols[0]}, medyan {vols[len(vols)//2]}, max {vols[-1]}")
    print(f"HÂLÂ boş (0 mL) vaka: {len(still_empty)}  {still_empty if still_empty else ''}")
    print("Çıktı:", os.path.join(OUT_ROOT, DATASET))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args().limit)
