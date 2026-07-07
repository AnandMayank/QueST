import json
import os

def save_metadata(metadata, out_root, obj_id):
    out_dir = os.path.join(out_root, obj_id)
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)