from pandas import read_csv

import torch


def get_labels_train(file_galaxy_labels) -> torch.Tensor:
    df_galaxy_labels = read_csv(file_galaxy_labels)
    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:]].values
    labels_train = torch.from_numpy(labels_train).float()
    return labels_train
