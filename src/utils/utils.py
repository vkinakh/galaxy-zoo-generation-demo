import numpy as np
import gdown

import torch


def download_model(file_id: str, output_path: str):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path)


def sample_labels(labels: torch.Tensor, n: int) -> torch.Tensor:
    high = labels.shape[0]
    idx = np.random.randint(0, high, size=n)
    return labels[idx]
