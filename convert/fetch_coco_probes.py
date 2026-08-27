# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Fetch the image probe set: 10 licence-clean COCO train2017 photos.

train2017, never val2017. val2017 is the blinded holdout and anything derived
from it inherits that status; a conversion check is not worth spending a
holdout on when the train split answers the same question.

Selection is deterministic -- sort by image_id, take the first ten of the
non-share-alike pool -- so this reproduces the same probes on any machine
rather than a different ten each run. `coco_probes_manifest.json` records what
that produced here; if a rerun disagrees, something upstream changed.

Licences kept: CC-BY 2.0, "No known copyright restrictions", US Government
Work. Share-alike is dropped as identifiable, per the upstream filter.

Source of the licence filter:
  https://github.com/weftspun/dataflow-coco-gemx
    coco_person_commercial_train2017/{images,licenses}.parquet
"""
import json, os, sys, urllib.request
import pandas as pd

META = sys.argv[1] if len(sys.argv) > 1 else "coco_person_commercial_train2017"
OUT = sys.argv[2] if len(sys.argv) > 2 else "coco_probes"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

lic = pd.read_parquet(f"{META}/licenses.parquet")
img = pd.read_parquet(f"{META}/images.parquet")
share_alike = set(lic[lic.share_alike]["license_id"])
keep = img[~img.license_id.isin(share_alike)].sort_values("image_id").head(N)

os.makedirs(OUT, exist_ok=True)
names = {r.license_id: r["name"] for _, r in lic.iterrows()}
manifest = []
for _, r in keep.iterrows():
    path = os.path.join(OUT, r.file_name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(r.coco_url, path)
    manifest.append({"image_id": int(r.image_id), "file": r.file_name,
                     "license_id": int(r.license_id), "w": int(r.width), "h": int(r.height)})
    print(f"  {r.file_name}  {r.width}x{r.height}  {names[r.license_id]}")
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"\n  {len(manifest)} probes -> {OUT}/")
