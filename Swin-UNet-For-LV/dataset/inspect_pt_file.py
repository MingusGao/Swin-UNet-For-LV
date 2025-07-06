import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm


def inspect_sample(pt_path: str):

    file_path = Path(pt_path)
    if not file_path.exists():
        print(f"[ERROR] 文件未找到: {pt_path}")
        return

    print(f"[INFO] 正在加载样本: {file_path.name}")
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"[ERROR] 加载文件失败. 原因: {e}")
        return

    clip = data.get('clip')
    mask = data.get('mask')
    mask_avail = data.get('mask_avail')
    ef = data.get('ef')

    if clip is None or mask is None or mask_avail is None:
        print("[ERROR] .pt 文件中缺少必要的键 ('clip', 'mask', 'mask_avail').")
        return

    print(f"[INFO] 视频切片包含 {clip.shape[0]} 帧。")
    if ef is not None:
        print(f"[INFO] 对应的 EF 值为: {ef.item():.2f}")

    output_dir = file_path.parent / "inspected" / file_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 正在生成帧图像...")
    for i in tqdm(range(clip.shape[0]), desc="生成帧"):
        frame_np = clip[i].permute(1, 2, 0).numpy() * 255
        frame_np = frame_np.astype('uint8')


        mask_np = mask[i].permute(1, 2, 0).numpy() * 255
        mask_np = mask_np.astype('uint8')
        mask_overlay_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
        mask_overlay_rgb[:, :, 1] = 0
        mask_overlay_rgb[:, :, 2] = 0
        

        
        overlayed_frame = cv2.addWeighted(frame_np, 0.8, mask_overlay_rgb, 0.6, 0)
        
        text = f"Frame: {i}"
        if mask_avail[i]:
            text += " (HAS MASK)"
        cv2.putText(overlayed_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        frame_filename = output_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(frame_filename), cv2.cvtColor(overlayed_frame, cv2.COLOR_RGB2BGR))

    print(f"\n[SUCCESS] 所有帧图像已保存至: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a pre-processed .pt sample file and save its frames as images.")
    parser.add_argument("--path", type=str, required=True, help="Path to the .pt file you want to inspect.")
    args = parser.parse_args()
    inspect_sample(args.path)
