import argparse
import torch
from pathlib import Path
import time
import numpy as np


from models.swin_unet import SwinUnetLV
from thop import profile


def profile_model(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not Path(args.ckpt).exists():
        print(f"[ERROR] Checkpoint file not found: {args.ckpt}")
        return

    print(f"[INFO] Loading model from: {args.ckpt}")
    model = SwinUnetLV(img_size=args.image_size, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.to(device).eval()


    num_frames_in_chunk = 32
    dummy_input = torch.randn(1, num_frames_in_chunk, 3, args.image_size, args.image_size).to(device)
    print(f"[INFO] Using a dummy input of shape: {list(dummy_input.shape)}")


    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    params_m = params / 1e6  # 百万 (Million)
    flops_g = flops / 1e9  # 十亿 (Giga)

    print("\n" + "=" * 50)
    print("--- Model Complexity Analysis ---")
    print(f"Total Parameters: {params_m:.2f} M")
    print(f"Total FLOPs (Multiply-Adds): {flops_g:.2f} GFLOPs")
    print("=" * 50)


    if device.type == 'cuda':

        print("\n--- GPU Performance Analysis ---")


        print("[INFO] Warming up the GPU...")
        for _ in range(10):
            _ = model(dummy_input)

        torch.cuda.synchronize()  


        timings = []
        print("[INFO] Measuring inference speed...")
        with torch.no_grad():
            for _ in range(args.num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)

        avg_latency = np.mean(timings) * 1000 


        fps = num_frames_in_chunk / np.mean(timings)


        print(f"Average Latency (for a {num_frames_in_chunk}-frame chunk): {avg_latency:.2f} ms")
        print(f"Inference Speed (Frames Per Second): {fps:.2f} FPS")


        torch.cuda.reset_peak_memory_stats(device)
        _ = model(dummy_input)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"Peak VRAM Usage: {peak_memory_mb:.2f} MB")

    else:

        print("\n--- CPU Performance Analysis ---")
        print("[INFO] Measuring inference speed...")
        timings = []
        with torch.no_grad():
            for _ in range(args.num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                timings.append(end_time - start_time)

        avg_latency = np.mean(timings) * 1000  

        fps = num_frames_in_chunk / np.mean(timings)


        print(f"Average Latency (for a {num_frames_in_chunk}-frame chunk): {avg_latency:.2f} ms")
        print(f"Inference Speed (Frames Per Second): {fps:.2f} FPS")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a trained model for computational resource usage.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained model checkpoint file.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size used during model training.")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of inference runs to average for speed measurement.")

    args = parser.parse_args()
    profile_model(args)
