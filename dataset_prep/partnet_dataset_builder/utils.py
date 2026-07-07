import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PARTNET_MOBILITY_ROOT", "<path-to-partnet-mobility-dataset>"))
OUT = Path("laptop_ids.txt")

laptops = []

for obj_dir in sorted(ROOT.iterdir()):
    meta = obj_dir / "meta.json"
    if not meta.exists():
        continue

    with open(meta) as f:
        data = json.load(f)

    if data.get("model_cat", "").lower() == "laptop":
        laptops.append(obj_dir.name)

OUT.write_text("\n".join(laptops))
print(f"Saved {len(laptops)} laptop IDs to {OUT}")
