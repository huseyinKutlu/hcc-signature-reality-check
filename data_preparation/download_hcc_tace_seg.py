"""
Adım 1A (düzeltilmiş) — HCC-TACE-Seg: önce SADECE SEG maskelerini indir
=======================================================================
Düzeltme: downloadSeries DataFrame için input_type="df" ister.
Strateji: 572 CT'nin hepsini DEĞİL, önce 105 SEG'i indir; sonra (seg_inspect.py)
her SEG'in referans aldığı CT'yi bulup yalnızca onları indir -> ~470 gereksiz
seri inmez.
"""
from tcia_utils import nbia

DEST = "/home/hkutlu/Desktop/JILTI/HCC-TACE-Seg"      # kendi diskinize göre değiştirin

series_df = nbia.getSeries(collection="HCC-TACE-Seg", format="df")
print("Modaliteler:", series_df["Modality"].value_counts().to_dict())

# yalnızca SEG maskeleri (küçük, hızlı)
seg_df = series_df[series_df["Modality"] == "SEG"]
print("İndirilecek SEG serisi:", len(seg_df))

nbia.downloadSeries(seg_df, input_type="df", path=DEST, max_workers=8)
print("SEG maskeleri indirildi ->", DEST)
print("Sıradaki: python seg_inspect.py  (segment adları + gerekli CT listesi)")

# Tüm seri listesini sonraki adım için sakla
series_df.to_csv("hcc_all_series.csv", index=False)
