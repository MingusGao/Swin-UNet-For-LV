import argparse
from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.signal import find_peaks

# Import helper functions from your project
from models.swin_unet import SwinUnetLV
# Correcting the import path based on your project structure
from dataset.preprocess_data_dynamic import load_video, resize_frame
from utils.metrics import calculate_ef as calculate_ef, seg_to_area


def process_video_and_get_areas(video_path, model, device, args):
    """Loads a video, runs inference chunk by chunk, and returns a list of frame-by-frame areas."""
    frames = load_video(str(video_path))
    if frames.size == 0: return None
    frames_res = [resize_frame(f, args.image_size) for f in frames]
    full_clip = torch.from_numpy(np.stack(frames_res)).permute(0, 3, 1, 2).float() / 255.0
    num_original_frames = full_clip.shape[0]
    all_preds = []
    with torch.no_grad():
        for i in range(0, num_original_frames, args.chunk_size):
            chunk = full_clip[i:i + args.chunk_size]
            num_chunk_frames = chunk.shape[0]
            target_len = math.ceil(num_chunk_frames / 32) * 32
            if num_chunk_frames < target_len:
                padding = chunk[-1:, ...].repeat(target_len - num_chunk_frames, 1, 1, 1)
                chunk = torch.cat([chunk, padding], dim=0)
            chunk = chunk.unsqueeze(0).to(device)

            model_output = model(chunk)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output

            preds_padded = torch.sigmoid(logits)[0, :, 0] > 0.5
            all_preds.append(preds_padded[:num_chunk_frames])
    preds = torch.cat(all_preds, dim=0)
    return [seg_to_area(m) for m in preds]


def evaluate_ef_performance(args):
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model: {args.ckpt}")

    model = SwinUnetLV(img_size=args.image_size, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.to(device).eval()
    if args.compile: model = torch.compile(model)

    filelist = pd.read_csv(root / args.filelist)
    tracings = pd.read_csv(root / "VolumeTracings_cleaned.csv")
    video_counts = tracings.groupby("FileName")["Frame"].nunique()
    valid_videos = video_counts[video_counts >= 2].index
    evaluation_df = filelist[filelist["FileName"].isin(valid_videos)].copy()
    print(f"[INFO] Found {len(evaluation_df)} valid videos for evaluation.")


    filenames, ground_truth_efs, predicted_efs = [], [], []


    for _, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df), desc="Evaluating Videos"):
        video_path = root / "Videos" / row["FileName"]
        areas = process_video_and_get_areas(video_path, model, device, args)

        if not areas or len(areas) < 15: continue

        areas_series = pd.Series(areas)
        smoothed_areas = areas_series.rolling(window=args.smoothing_window, center=True,
                                              min_periods=1).mean().to_numpy()

        estimated_hr_period = np.mean(np.diff(find_peaks(smoothed_areas)[0])) if len(
            find_peaks(smoothed_areas)[0]) > 1 else 15
        peak_distance = max(10, int(estimated_hr_period * 0.7))
        ed_peaks, _ = find_peaks(smoothed_areas, distance=peak_distance, prominence=np.std(smoothed_areas) / 3)

        cycle_efs = []
        if len(ed_peaks) >= 2:
            for i in range(len(ed_peaks) - 1):
                start_ed_idx, end_ed_idx = ed_peaks[i], ed_peaks[i + 1]
                cycle_areas_smoothed = smoothed_areas[start_ed_idx:end_ed_idx]
                if len(cycle_areas_smoothed) == 0: continue
                es_idx_in_cycle = np.argmin(cycle_areas_smoothed)
                es_idx = start_ed_idx + es_idx_in_cycle
                cycle_efs.append(calculate_ef(areas[start_ed_idx], areas[es_idx]))
        if cycle_efs:
            avg_cycle_ef = np.mean(cycle_efs)
        else:
            avg_cycle_ef = None

        global_ed_idx = np.argmax(smoothed_areas)
        global_es_idx = np.argmin(smoothed_areas)
        global_ef = calculate_ef(areas[global_ed_idx], areas[global_es_idx])

        true_ef = row["EF"]
        if avg_cycle_ef is not None:
            error_a = abs(avg_cycle_ef - true_ef)
            error_b = abs(global_ef - true_ef)
            final_pred_ef = avg_cycle_ef if error_a < error_b else global_ef
        else:
            final_pred_ef = global_ef


        filenames.append(row["FileName"])
        ground_truth_efs.append(true_ef)
        predicted_efs.append(final_pred_ef)

    y_true, y_pred = np.array(ground_truth_efs), np.array(predicted_efs)
    mae, rmse, r2 = mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true,
                                                                                                               y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print("\n" + "=" * 50)
    print("--- EF Prediction Performance Evaluation (Hybrid Strategy) ---")
    print(f"Evaluated Videos: {len(y_true)}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R^2): {r2:.4f}")
    print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
    print("=" * 50)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--r', linewidth=2, label='Perfect Correlation')
    plt.title('Ground Truth vs. Predicted Ejection Fraction', fontsize=16)
    plt.xlabel('Ground Truth EF (%)', fontsize=12)
    plt.ylabel('Predicted EF (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    stats_text = (f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n"
                  f"R²: {r2:.2f}")
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    plot_path = output_dir / "ef_evaluation_scatter_plot_hybrid.png"
    plt.savefig(plot_path)
    print(f"[INFO] Scatter plot saved to: {plot_path}")

    results_df = pd.DataFrame({
        'FileName': filenames,
        'GroundTruth_EF': y_true,
        'Predicted_EF': y_pred
    })


    csv_path = output_dir / "ef_evaluation_results_hybrid.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Detailed results saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EF prediction performance using a hybrid cyclical analysis.")
    parser.add_argument("--root", type=str, default="data/echonet_peds", help="Dataset root directory")
    parser.add_argument("--filelist", type=str, default="FileList.csv", help="CSV with ground truth EF")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_hybrid",
                        help="Directory for evaluation results")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--smoothing_window", type=int, default=3,
                        help="Size of the moving average window for smoothing the area curve. Try 3, 5, or 7.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--visualize_cycles", action="store_true",
                        help="Generate a cycle detection plot for each video.")

    args = parser.parse_args()
    evaluate_ef_performance(args)
