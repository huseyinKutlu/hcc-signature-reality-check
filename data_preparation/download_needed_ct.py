"""
Adım 1C — Yalnızca SEG'lerin referans aldığı CT serilerini indir (104 seri, 572 değil)
=====================================================================================
seg_inspect.py'nin ürettiği needed_ct_uids.csv + hcc_all_series.csv kullanılır.
SEG'ler zaten aynı DEST'te; tcia_utils indirilmişleri atlar.
"""
import pandas as pd
from tcia_utils import nbia

DEST = "/home/hkutlu/Desktop/JILTI/HCC-TACE-Seg"

alls = pd.read_csv("hcc_all_series.csv")
need = pd.read_csv("needed_ct_uids.csv")
ct = alls[alls["SeriesInstanceUID"].isin(need["SeriesInstanceUID"])].copy()
print("İndirilecek CT serisi:", len(ct), "(tüm 572 yerine)")

nbia.downloadSeries(ct, input_type="df", path=DEST, max_workers=8)
print("Gerekli CT serileri indirildi ->", DEST)
print("Sıradaki: python prepare_nnunet_dataset.py --limit 3   (önce 3 vaka test)")
