import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class PianoRollDataset(Dataset):
    def __init__(self, metadata_csv, root_dir, transform=None, dtype=torch.float32):
        self.meta = pd.read_csv(metadata_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Generate filename from row index
        filename = f"metal_{idx:07d}.npy"  # zero-padded to match your file names
        file_path = os.path.join(self.root_dir, filename)

        # Load .npy piano-roll
        roll = np.load(file_path)  # no pickle needed if it's a standard array
        x = torch.tensor(roll, dtype=self.dtype)

        # Normalize if values > 1 (velocity piano-roll)
        if x.max() > 1:
            x = x / 127.0

        if self.transform:
            x = self.transform(x)

        # Return tensor + metadata as dict
        row_meta = self.meta.iloc[idx].to_dict()
        return x, row_meta



# Wrap all code that starts DataLoader in main guard for Windows
if __name__ == "__main__":
    root_dir = r"C:\Projects\Music_ML_Pr\music-ml-app\data\midi\metal\processed\arrays"
    metadata_csv = r"C:\Projects\Music_ML_Pr\music-ml-app\data\midi\metal\processed\metal_metadata.csv"

    dataset = PianoRollDataset(metadata_csv, root_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=False)
    
    for batch, meta in dataloader:
        print("Batch shape:", batch.shape)
        first_row_meta = {k: v[0] for k, v in meta.items()}
        print("Metadata for first sample:", first_row_meta)
        break
