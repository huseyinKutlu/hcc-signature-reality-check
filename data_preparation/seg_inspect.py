"""
Adım 1B — İndirilen SEG'leri incele: segment adları + referans CT serileri
==========================================================================
Çıktı:
  - Hangi segment etiketleri var ve kaç SEG'de (tümör segmentinin GERÇEK adı!)
  - Kaç SEG dosyası var (=hasta sayısı beklenir)
  - needed_ct_uids.csv : indirilecek CT serilerinin UID'leri (sadece bunlar!)
"""
import glob, os, csv
from collections import Counter
import pydicom

DEST = "/home/hkutlu/Desktop/JILTI/HCC-TACE-Seg"      # download scriptindeki ile aynı

rows, needed_ct = [], set()
label_freq = Counter()

for f in glob.glob(os.path.join(DEST, "**", "*.dcm"), recursive=True):
    try:
        d = pydicom.dcmread(f, stop_before_pixels=True)
    except Exception:
        continue
    if getattr(d, "Modality", "") != "SEG":
        continue
    labels = [str(getattr(s, "SegmentLabel", f"seg{i+1}"))
              for i, s in enumerate(getattr(d, "SegmentSequence", []))]
    for l in labels:
        label_freq[l] += 1
    pid = getattr(d, "PatientID", "?")
    ref = ""
    try:
        ref = d.ReferencedSeriesSequence[0].SeriesInstanceUID
        needed_ct.add(ref)
    except Exception:
        pass
    rows.append((pid, "|".join(labels), ref))

print("=== SEG ENVANTERİ ===")
print("SEG dosyası (hasta):", len(rows))
print("Segment etiketleri (ad -> kaç SEG'de):")
for lab, n in label_freq.most_common():
    print(f"   '{lab}': {n}")
print("Referans CT serisi (indirilecek benzersiz):", len(needed_ct))
no_ref = sum(1 for r in rows if not r[2])
if no_ref:
    print(f"[UYARI] {no_ref} SEG'in referans CT'si okunamadı")

with open("needed_ct_uids.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["SeriesInstanceUID"])
    for u in sorted(needed_ct):
        w.writerow([u])
print("needed_ct_uids.csv yazıldı ->", len(needed_ct), "CT UID")
