import numpy as np
import torch
from torch.utils.data import Dataset


class WeeklyCrashDataset(Dataset):
    def __init__(self, features, feature_cols, sample_dates, labels, seq_len: int = 60):
        data = features[feature_cols].values.astype(np.float32)
        label_values = labels.reindex(sample_dates).values.astype(np.float32)
        xs, ys, ds = [], [], []
        for i, day in enumerate(sample_dates):
            pos = features.index.searchsorted(day, side="right") - 1
            if pos < seq_len - 1:
                continue
            xs.append(data[pos - seq_len + 1 : pos + 1])
            ys.append(label_values[i])
            ds.append(day)
        self.x = torch.tensor(np.array(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.float32)
        self.dates = ds

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
