from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


class EchoNetPedsDataset_Optimized(Dataset):


    def __init__(self, root: str, split: str = "train", filelist_name: str = "Processed_FileList.csv"):
        self.root = Path(root)


        if "single_frame" in filelist_name:
            self.processed_dir = self.root / "processed_single_frame"
        else:
            self.processed_dir = self.root / "processed_dynamic"

        metadata_path = self.root / filelist_name
        try:
            full_metadata = pd.read_csv(metadata_path)
        except FileNotFoundError:
            print(f"[ERROR] Processed metadata file not found: {metadata_path}")
            print("Please run preprocess_data.py first to generate the pre-processed data.")
            raise

        if split == "train":
            self.metadata = full_metadata[full_metadata["Split"] <= 7]
        elif split == "val":
            self.metadata = full_metadata[full_metadata["Split"] == 8]
        else:  # test
            self.metadata = full_metadata[full_metadata["Split"] >= 9]

 
        if "SampleFileName" in self.metadata.columns:
            self.path_list = self.metadata["SampleFileName"].tolist()
        elif "SamplePath" in self.metadata.columns:
            self.path_list = self.metadata["SamplePath"].tolist()
        else:
            raise KeyError("Could not find 'SampleFileName' or 'SamplePath' in the provided metadata CSV file.")

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        path_info = self.path_list[i]

        
        sample_filename = Path(path_info).name


        sample_path = self.processed_dir / sample_filename
       

        data = torch.load(sample_path)
        return data['clip'], data['mask'], data['mask_avail'], data['ef']
