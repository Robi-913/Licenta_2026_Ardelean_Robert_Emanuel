# save as check_distribution.py
import json

master_json = "data/oct5k/metadata/_master.json"

BIOMARKERS = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]

def normalize_class(cls):
    return cls.lower().replace(" ", "").replace("_", "")

BM_NORMALIZED = {normalize_class(b): b for b in BIOMARKERS}

with open(master_json, "r", encoding="utf-8") as f:
    master = json.load(f)

counts = {bm: 0 for bm in BIOMARKERS}
total = 0

for meta in master:
    if not meta.get("has_bounding_boxes"):
        continue
    total += 1
    seen = set()
    for les in meta.get("lesions", []):
        cls_norm = normalize_class(les.get("class", ""))
        bm_key = BM_NORMALIZED.get(cls_norm)
        if bm_key and bm_key not in seen:
            counts[bm_key] += 1
            seen.add(bm_key)

print(f"\nTotal imagini cu bbox: {total}")
print(f"\nDistributie biomarkeri:")
for bm, count in sorted(counts.items(), key=lambda x: -x[1]):
    pct = 100 * count / total if total > 0 else 0
    print(f"  {bm:<25}: {count:>4} ({pct:.1f}%)")