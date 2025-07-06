import argparse
import random
from pathlib import Path
import math
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models.swin_unet import SwinUnetLV


def load_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames)

def resize_frame(frame: np.ndarray, size: int = 224) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

def seg_to_area(mask: torch.Tensor) -> float:
    return mask.sum().item()

def calculate_ef(area_ed: float, area_es: float) -> float:
    if area_ed == 0:
        return 0.0
    ef = ((area_ed - area_es) / area_ed) * 100.0
    return max(0, min(ef, 100.0))


def process_single_video(video_path: Path, model, device, output_dir: Path, image_size: int, chunk_size: int = 32):

    try:

        frames = load_video(str(video_path))
        if frames.size == 0:
            print(f" [WARNING] Skipping empty or corrupted video: {video_path.name}")
            return

        frames_res = [resize_frame(f, image_size) for f in frames]
        full_clip = torch.from_numpy(np.stack(frames_res)).permute(0, 3, 1, 2).float() / 255.0
        num_original_frames = full_clip.shape[0]

        all_preds = []
        with torch.no_grad():
            for i in range(0, num_original_frames, chunk_size):
                chunk = full_clip[i:i + chunk_size]

                num_chunk_frames = chunk.shape[0]
                target_len = math.ceil(num_chunk_frames / 32) * 32
                if num_chunk_frames < target_len:
                    padding_needed = target_len - num_chunk_frames
                    last_frame = chunk[-1:, ...]
                    padding = last_frame.repeat(padding_needed, 1, 1, 1)
                    chunk = torch.cat([chunk, padding], dim=0)

                chunk = chunk.unsqueeze(0).to(device) # (1, T_padded, C, H, W)

                logits = model(chunk)
                preds_padded = torch.sigmoid(logits)[0, :, 0] > 0.5

                all_preds.append(preds_padded[:num_chunk_frames])


        preds = torch.cat(all_preds, dim=0)


        areas = [seg_to_area(m) for m in preds]
        ed_idx = int(np.argmax(areas))
        es_idx = int(np.argmin(areas))
        ef_pred = calculate_ef(areas[ed_idx], areas[es_idx])
        print(f"  - Predicted EF: {ef_pred:.2f}% (ED Frame: {ed_idx}, ES Frame: {es_idx})")


        output_path = output_dir / f"{video_path.stem}_overlay.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w = frames[0].shape[:2]
        cap.release()

        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for i, (original_frame, pred_mask) in enumerate(zip(frames, preds)):
            mask_rgb = (pred_mask.cpu().numpy() * 255).astype("uint8")
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2BGR)
            mask_resized = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
            green_mask = np.zeros_like(original_frame)
            green_mask[:, :, 1] = mask_resized[:, :, 1]
            overlay = cv2.addWeighted(original_frame, 0.8, green_mask, 0.5, 0)

            frame_text = f"Frame: {i}"
            if i == ed_idx: frame_text += " (ED)"
            elif i == es_idx: frame_text += " (ES)"
            cv2.putText(overlay, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        writer.release()

    except Exception as e:
        print(f" [ERROR] Failed to process {video_path.name}. Reason: {e}")



def main(args):

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading model from {args.ckpt}...")
    model = SwinUnetLV(img_size=args.image_size, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()
    if args.compile:
        print("[INFO] Compiling the model for extra speed...")
        model = torch.compile(model)

    all_videos = list(video_dir.glob("*.avi"))
    if not all_videos:
        print(f"[ERROR] No .avi videos found in {video_dir}")
        return

    num_to_select = min(args.num_videos, len(all_videos))
    selected_videos = random.sample(all_videos, num_to_select)
    print(f"[INFO] Found {len(all_videos)} videos. Randomly selected {num_to_select} for inference.")
    print(f"[INFO] Output will be saved to: {output_dir}")

    for video_path in tqdm(selected_videos, desc="Batch Processing Videos"):
        print(f"\n--- Processing {video_path.name} ---")
        process_single_video(
            video_path=video_path,
            model=model,
            device=device,
            output_dir=output_dir,
            image_size=args.image_size,
            chunk_size=args.chunk_size
        )

    print("\n[SUCCESS] Batch inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on random videos from a directory.")
    parser.add_argument("--video_dir", type=str, default="data/echonet_peds/test_video", help="Directory containing input video files.")
    parser.add_argument("--output_dir", type=str, default="data/echonet_peds/mask", help="Directory to save the output videos.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="Path to the model checkpoint file.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size used during model training.")
    parser.add_argument("--num_videos", type=int, default=10, help="Number of random videos to process.")
    parser.add_argument("--chunk_size", type=int, default=32, help="Number of frames to process in one chunk to save VRAM.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for extra speed (requires PyTorch 2.0+).")

    args = parser.parse_args()
    main(args)
