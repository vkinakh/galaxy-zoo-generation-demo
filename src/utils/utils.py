import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

import torch


def download_model(file_id: str, output_path: str):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=output_path)


def sample_labels(labels: torch.Tensor, n: int) -> torch.Tensor:
    high = labels.shape[0]
    idx = np.random.randint(0, high, size=n)
    return labels[idx]
