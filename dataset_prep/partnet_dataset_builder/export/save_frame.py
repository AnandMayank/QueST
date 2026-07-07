import os
import imageio.v2 as imageio
import numpy as np

def save_frame(out_dir, idx, rgb, depth):
    os.makedirs(out_dir, exist_ok=True)
    
    # Handle different RGB value ranges from SAPIEN
    rgb = np.array(rgb, dtype=np.float32)
    
    # Check if values are in [0, 1] or [0, 255] range
    if rgb.max() <= 1.0:
        # Values in [0, 1] range - multiply by 255
        rgb_data = (rgb * 255).clip(0, 255).astype("uint8")
    else:
        # Values already in [0, 255] range (or higher) - just clip
        rgb_data = rgb.clip(0, 255).astype("uint8")

    imageio.imwrite(
        os.path.join(out_dir, f"{idx:05d}.png"),
        rgb_data,
    )

    np.save(
        os.path.join(out_dir, f"{idx:05d}_depth.npy"),
        depth.astype("float32"),
    )