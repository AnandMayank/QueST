import os
import imageio.v2 as imageio

def frames_to_video(frames_dir, out_path, fps=30):
    images = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    )

    if not images:
        print(f"Warning: No frames found in {frames_dir}")
        return

    try:
        writer = imageio.get_writer(out_path, fps=fps)
        for img in images:
            writer.append_data(imageio.imread(os.path.join(frames_dir, img)))
        writer.close()
    except Exception as e:
        print(f"[ERROR] Video creation failed (likely missing ffmpeg): {e}")