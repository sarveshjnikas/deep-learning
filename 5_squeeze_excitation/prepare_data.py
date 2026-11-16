# BUILDS data/{train,val,test}/<class>/<file>.tif FROM THE OFFICIAL UC MERCED DOWNLOAD,
# USING THE EXACT SPLIT IN split.csv (70/10/20 IMAGES PER CLASS) SO THE NUMBERS IN THE README REPRODUCE
#
#   python prepare_data.py /path/to/UCMerced_LandUse/Images

import csv, shutil, sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "data")

with open("split.csv") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    out = dst / row["split"] / row["class"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / row["class"] / row["file"], out / row["file"])

counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
print(f"copied {len(rows)} images to {dst}/ ->", counts)
