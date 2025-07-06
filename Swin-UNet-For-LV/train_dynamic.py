import argparse
import random
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.echonet_peds import EchoNetPedsDataset_Optimized as EchoNetPedsDataset
from models.swin_unet import SwinUnetLV
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.losses import CombinedLoss
from utils.metrics import calculate_metrics


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg_path: str):
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    warmup_epochs = cfg.get("warmup_epochs", 3)
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    
    processed_filelist = cfg.get("processed_filelist")
    print(f"[INFO] Loading pre-processed data using metadata: {processed_filelist}")

    train_ds = EchoNetPedsDataset(root=cfg["dataset_root"], split="train", filelist_name=processed_filelist)
    val_ds = EchoNetPedsDataset(root=cfg["dataset_root"], split="val", filelist_name=processed_filelist)
    test_ds = EchoNetPedsDataset(root=cfg["dataset_root"], split="test", filelist_name=processed_filelist)

    print(f"[INFO] Train/Val/Test size = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers", 0),
                              pin_memory=True if cfg.get("num_workers", 0) > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    
    model = SwinUnetLV(img_size=cfg["image_size"], pretrained=False).to(device)

    initial_ckpt_path = cfg.get("initial_ckpt_path")
    if initial_ckpt_path and Path(initial_ckpt_path).exists():
        try:
            model.load_state_dict(torch.load(initial_ckpt_path, map_location=device))
            print(f"[INFO] Successfully loaded initial weights from: {initial_ckpt_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load initial weights. Starting from scratch. Reason: {e}")
    else:
        print("[WARNING] Initial checkpoint path not found or file does not exist. Starting from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get("amp", True))
    main_scheduler = CosineAnnealingLR(optimizer, T_max=(cfg["num_epochs"] - warmup_epochs) * len(train_loader),
                                       eta_min=1e-7)

    
    best_val_dice = 0.0
    best_ckpt_path = None
    train_loss_history, val_loss_history, val_dice_history, val_iou_history = [], [], [], []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        running_loss, valid_batches = 0.0, 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Finetuning")

        for i, (clips, masks, m_avail, _) in pbar:
            if epoch < warmup_epochs:
                lr_scale = (i + 1 + epoch * len(train_loader)) / (warmup_epochs * len(train_loader))
                current_lr = lr * lr_scale
                for param_group in optimizer.param_groups: param_group['lr'] = current_lr
            else:
                current_lr = main_scheduler.get_last_lr()[0]

            clips, masks, m_avail = clips.to(device), masks.to(device), m_avail.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=cfg.get("amp", True)):
                logits = model(clips)
                loss = criterion(logits, masks, m_avail)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if epoch >= warmup_epochs: main_scheduler.step()
            if loss.item() > 0:
                running_loss += loss.item()
                valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        train_loss_avg = running_loss / valid_batches if valid_batches > 0 else 0.0
        print(f"\nEpoch {epoch}: Train Loss {train_loss_avg:.4f}")
        train_loss_history.append(train_loss_avg)

        
        model.eval()
        val_running_loss, val_total_dice, val_total_iou, val_valid_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for clips, masks, m_avail, _ in val_loader:
                clips, masks, m_avail = clips.to(device), masks.to(device), m_avail.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=cfg.get("amp", True)):
                    logits = model(clips)
                    loss = criterion(logits, masks, m_avail)
                    dice, iou = calculate_metrics(logits, masks, m_avail)
                if loss.item() > 0:
                    val_running_loss += loss.item()
                    if dice is not None:
                        val_total_dice += dice
                        val_total_iou += iou
                    val_valid_batches += 1

        val_loss_avg = val_running_loss / val_valid_batches if val_valid_batches > 0 else 0.0
        val_dice_avg = val_total_dice / val_valid_batches if val_valid_batches > 0 else 0.0
        val_iou_avg = val_total_iou / val_valid_batches if val_valid_batches > 0 else 0.0
        print(f"Epoch {epoch}: Validation Loss {val_loss_avg:.4f} | Dice: {val_dice_avg:.4f} | IoU: {val_iou_avg:.4f}")
        val_loss_history.append(val_loss_avg)
        val_dice_history.append(val_dice_avg)
        val_iou_history.append(val_iou_avg)

        
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            ckpt_path = save_dir / "best_model_finetuned.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_ckpt_path = ckpt_path
            print(f"[✓] Saved better model to {ckpt_path} (Validation Dice: {best_val_dice:.4f})")

    
    print("\n" + "=" * 50)
    print("Finetuning complete.")

    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(train_loss_history, label="Train Loss", marker='o')
    ax1.plot(val_loss_history, label="Validation Loss", marker='o')
    ax1.set_title("Training & Validation Loss Curve", fontsize=16)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    ax2.plot(val_dice_history, label="Validation Dice Score", marker='o', color='g')
    ax2.plot(val_iou_history, label="Validation IoU", marker='o', color='r')
    ax2.set_title("Validation Metrics Curve", fontsize=16)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path)
    print(f"[✓] Training curves saved to: {plot_path}")

    # ---- Evaluate the best model on the test set ----
    print("\n" + "=" * 50)
    print("Evaluating the best model on the test set...")

    if best_ckpt_path is not None:
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[INFO] Successfully loaded best model: {best_ckpt_path}")

        model.eval()
        test_loss, test_dice, test_iou, test_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for clips, masks, m_avail, _ in tqdm(test_loader, desc="Testing"):
                clips, masks, m_avail = clips.to(device), masks.to(device), m_avail.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=cfg.get("amp", True)):
                    logits = model(clips)
                    loss = criterion(logits, masks, m_avail)
                    dice, iou = calculate_metrics(logits, masks, m_avail)

                if loss.item() > 0:
                    test_loss += loss.item()
                    if dice is not None:
                        test_dice += dice
                        test_iou += iou
                    test_batches += 1

        test_loss_avg = test_loss / test_batches if test_batches > 0 else 0.0
        test_dice_avg = test_dice / test_batches if test_batches > 0 else 0.0
        test_iou_avg = test_iou / test_batches if test_batches > 0 else 0.0

        print("\n--- [ FINAL TEST RESULTS ] ---")
        print(f"Test Loss:      {test_loss_avg:.4f}")
        print(f"Test Dice Score:  {test_dice_avg:.4f}")
        print(f"Test IoU (Jaccard): {test_iou_avg:.4f}")

        
        results_path = save_dir / "test_results.txt"
        with open(results_path, 'w') as f:
            f.write("--- Final Test Set Evaluation Results ---\n")
            f.write(f"Best Model Checkpoint: {best_ckpt_path}\n\n")
            f.write(f"Test Loss: {test_loss_avg:.4f}\n")
            f.write(f"Test Dice Score: {test_dice_avg:.4f}\n")
            f.write(f"Test IoU (Jaccard): {test_iou_avg:.4f}\n")
        print(f"Test results saved to: {results_path}")
     

    else:
        print("[WARNING] No best model was saved. Skipping test set evaluation.")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the config file.")
    args = parser.parse_args()
    main(args.config)
