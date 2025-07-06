import argparse
from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames)


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def polygon_to_mask(pts: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def _window_indices(T: int, center: int, window: int) -> list:
    beg = max(center - window, 0)
    end = min(center + window, T - 1)
    idx = list(range(beg, end + 1))
    while len(idx) < 2 * window + 1:
        if idx[0] > 0:
            idx.insert(0, idx[0] - 1)
        else:
            idx.append(min(idx[-1] + 1, T - 1))
    return idx


def preprocess_dataset(args):
    root = Path(args.root)
    output_dir = root / "processed_dynamic"  
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Pre-processed data will be saved to: {output_dir}")
    print(f"[INFO] Using a context window size of: {args.window}")

    filelist = pd.read_csv(root / args.filelist)
    tracings = pd.read_csv(root / "VolumeTracings_cleaned.csv")

    original_size = 112
    scale_factor = args.image_size / original_size

    processed_tracings = []
    for vid, df in tqdm(tracings.groupby("FileName"), desc="Processing Tracings"):
        pts_by_f = {}
        for f, g in df.groupby("Frame"):
            pts = g[["X", "Y"]].values.astype(np.float32)
            if pts.shape[0] >= 3:
                scaled_pts = pts * scale_factor
                pts_by_f[int(f)] = scaled_pts
        if len(pts_by_f) < 2: continue
        areas = {f: cv2.contourArea(p) for f, p in pts_by_f.items()}
        ed_frame = max(areas, key=areas.get)
        es_frame = min(areas, key=areas.get)
        masks = {f: polygon_to_mask(pts_by_f[f], args.image_size, args.image_size) for f in (ed_frame, es_frame)}
        processed_tracings.append({"FileName": vid, "ed_frame": ed_frame, "es_frame": es_frame, "masks": masks})

    tracings_summary = pd.DataFrame(processed_tracings)
    merged_data = pd.merge(filelist, tracings_summary, on="FileName", how="inner")
    print(f"[INFO] Found {len(merged_data)} valid videos to process.")

    processed_samples_metadata = []
    for _, row in tqdm(merged_data.iterrows(), total=len(merged_data), desc="Generating Samples"):
        vid = row["FileName"]
        video_path = root / "Videos" / vid
        clip_raw = load_video(str(video_path))
        if clip_raw.size == 0: continue

        md = row.to_dict()
        ed_f, es_f = md["ed_frame"], md["es_frame"]

        for center_frame_type, center_frame_idx in [("ED", ed_f), ("ES", es_f)]:
            idx = _window_indices(len(clip_raw), center_frame_idx, args.window)

            target_len = math.ceil(len(idx) / 32) * 32
            while len(idx) < target_len:
                for i in range(len(idx) - 2, -1, -1):
                    if len(idx) >= target_len: break
                    idx.append(idx[i])
                if len(idx) < target_len:
                    idx.append(idx[-1])

            clip = torch.from_numpy(np.stack([resize_frame(clip_raw[j], args.image_size) for j in idx])).permute(0, 3,
                                                                                                                 1,
                                                                                                                 2).float() / 255.0
            mask = torch.zeros((len(idx), 1, args.image_size, args.image_size), dtype=torch.float32)
            mask_avail = torch.zeros(len(idx), dtype=torch.bool)
            for f in (ed_f, es_f):
                if f in idx:
                    m = md["masks"][f]
                    t = idx.index(f)
                    mask[t, 0] = torch.from_numpy(m).float()
                    mask_avail[t] = True

            ef_val = torch.tensor(md["EF"], dtype=torch.float32)
            sample_data = {'clip': clip, 'mask': mask, 'mask_avail': mask_avail, 'ef': ef_val}
            sample_filename = f"{Path(vid).stem}_{center_frame_type}.pt"
            torch.save(sample_data, output_dir / sample_filename)
            processed_samples_metadata.append({"SamplePath": str(output_dir / sample_filename), "Split": row["Split"]})

    final_metadata_df = pd.DataFrame(processed_samples_metadata)
    metadata_path = root / f"Processed_dynamic_{args.filelist}"
    final_metadata_df.to_csv(metadata_path, index=False)
    print(f"[SUCCESS] Pre-processing complete. New metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process the EchoNet-Peds dataset for dynamic video fine-tuning.")
    parser.add_argument("--root", type=str, default="data/echonet_peds", help="Root directory of the dataset.")
    parser.add_argument("--filelist", type=str, default="FileList.csv",
                        help="Name of the file list CSV (e.g., FileList_mini.csv).")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--window", type=int, default=8, help="Number of context frames on each side of the keyframe.")
    args = parser.parse_args()
    preprocess_dataset(args)
