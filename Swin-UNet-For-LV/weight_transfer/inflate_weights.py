import argparse
import torch
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.swin_unet import SwinUnetLV


def transfer_weights(source_ckpt_path: str, save_path: str, image_size: int):

    print("--- Start transfer weight ---")

    source_path = Path(source_ckpt_path)
    if not source_path.exists():
        print(f"[ERROR]cannot find weight: {source_ckpt_path}")
        return

    device = torch.device("cpu")  
    
    print(f"[INFO] from {source_path.name} loading weight ...")
    pretrained_dict = torch.load(source_path, map_location=device)

    target_model = SwinUnetLV(img_size=image_size, pretrained=False)
    target_dict = target_model.state_dict()
    print("[INFO] create target model")


    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict}


    target_dict.update(pretrained_dict)

    target_model.load_state_dict(target_dict)

    print("[INFO] weight transfer ")


    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target_model.state_dict(), save_path_obj)

    print(f"\n[SUCCESS] save model to: {save_path_obj}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer weights from a model trained on keyframes to a new model for fine-tuning on video sequences.")
    parser.add_argument("--source_ckpt", type=str, required=True,
                        help="Path to the source checkpoint file (your trained model).")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the new initialized checkpoint file.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size of the model.")

    args = parser.parse_args()
    transfer_weights(args.source_ckpt, args.save_path, args.image_size)
