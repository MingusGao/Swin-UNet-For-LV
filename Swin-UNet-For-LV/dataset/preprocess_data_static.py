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



def preprocess_dataset(args):
    root = Path(args.root)
    output_dir = root / "processed_single_frame"  
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Pre-processed data will be saved to: {output_dir}")
    print(f"[INFO] Strategy: Training on KEYFRAMES ONLY (ED/ES).")

    filelist = pd.read_csv(root / args.filelist)
    tracings = pd.read_csv(root / "VolumeTracings_cleaned.csv")

    original_size = 112
    scale_factor = args.image_size / original_size

    processed_tracings = []
    for vid, df in tqdm(tracings.groupby("FileName"), desc="Processing Tracings"):
        pts_by_f = {}
        for f, g in df.groupby("Frame"):
            pts = g[["X", "Y"]].values.astype(np.float32)
            pts = pts[~np.isnan(pts).any(axis=1)]
            if pts.shape[0] >= 3:
                scaled_pts = pts * scale_factor
                pts_by_f[int(f)] = scaled_pts

        if len(pts_by_f) < 2: continue

        areas = {f: cv2.contourArea(p) for f, p in pts_by_f.items()}
        ed_frame_idx = max(areas, key=areas.get)
        es_frame_idx = min(areas, key=areas.get)

        ed_mask = polygon_to_mask(pts_by_f[ed_frame_idx], args.image_size, args.image_size)
        es_mask = polygon_to_mask(pts_by_f[es_frame_idx], args.image_size, args.image_size)

        processed_tracings.append({
            "FileName": vid,
            "ed_frame": ed_frame_idx,
            "es_frame": es_frame_idx,
            "ed_mask": ed_mask,
            "es_mask": es_mask
        })

    tracings_summary = pd.DataFrame(processed_tracings)
    merged_data = pd.merge(filelist, tracings_summary, on="FileName", how="inner")
    print(f"[INFO] Found {len(merged_data)} valid videos to process.")

    processed_samples_metadata = []
    for _, row in tqdm(merged_data.iterrows(), total=len(merged_data), desc="Generating Samples"):
        vid = row["FileName"]
        video_path = root / "Videos" / vid
        clip_raw = load_video(str(video_path))
        if clip_raw.size == 0: continue

        
        for frame_type in ["ED", "ES"]:
            frame_idx = row[f"{frame_type.lower()}_frame"]
            mask_np = row[f"{frame_type.lower()}_mask"]

            if frame_idx >= len(clip_raw):
                print(
                    f" [WARNING] Skipping frame {frame_idx} for video {vid} as it is out of bounds (video has {len(clip_raw)} frames).")
                continue

            key_frame_np = resize_frame(clip_raw[frame_idx], args.image_size)

            # (H, W, C) -> (C, H, W)
            key_frame_tensor = torch.from_numpy(key_frame_np).permute(2, 0, 1).float() / 255.0

            key_mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()  # (1, H, W)

            
            target_len = 32
            clip = key_frame_tensor.unsqueeze(0).repeat(target_len, 1, 1, 1)  # (32, C, H, W)
            mask = key_mask_tensor.unsqueeze(0).repeat(target_len, 1, 1, 1)  # (32, 1, H, W)
            
            mask_avail = torch.ones(target_len, dtype=torch.bool)

            ef_val = torch.tensor(row["EF"], dtype=torch.float32)

            sample_data = {'clip': clip, 'mask': mask, 'mask_avail': mask_avail, 'ef': ef_val}
            sample_filename = f"{Path(vid).stem}_{frame_type}.pt"
            torch.save(sample_data, output_dir / sample_filename)

            processed_samples_metadata.append({
                "SamplePath": str(output_dir / sample_filename),
                "Split": row["Split"]
            })

    final_metadata_df = pd.DataFrame(processed_samples_metadata)
    metadata_path = root / f"Processed_{args.filelist}"
    final_metadata_df.to_csv(metadata_path, index=False)
    print(f"[SUCCESS] Pre-processing complete. New metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process dataset using KEYFRAMES ONLY.")
    parser.add_argument("--root", type=str, default="data/echonet_peds", help="Root directory of the dataset.")
    parser.add_argument("--filelist", type=str, default="FileList.csv",
                        help="Name of the file list CSV (e.g., FileList.csv or FileList_mini.csv).")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--window", type=int, default=0, help="This argument is ignored in this script version.")
    args = parser.parse_args()
    preprocess_dataset(args)
