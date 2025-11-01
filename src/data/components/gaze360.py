from pathlib import Path
from typing import Callable

import h5py
import torch
from torch.utils.data import ConcatDataset, Dataset

SUBJECT = {
    "train": 0,
    "val": 1,
    "test": 2,
    "unused": 3,
}


class OneFoldGaze360(Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Callable,
        mode: str,  # ['train', 'val', 'test', 'unused'].
        with_subject: bool = False,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.with_subject = with_subject
        if self.with_subject:
            self.subject = SUBJECT[mode]

    def __len__(self):
        with h5py.File(self.data_dir, "r", swmr=True) as f:
            return len(f[self.mode]["face_patch"])

    def __getitem__(self, idx):
        with h5py.File(self.data_dir, "r", swmr=True) as f:
            image = f[self.mode]["face_patch"][idx]
            gaze = f[self.mode]["face_gaze"][idx].astype("float")
        image = self.transform(image)
        # pose = torch.from_numpy(pose)
        gaze = torch.from_numpy(gaze)
        if self.with_subject:
            return image, gaze, self.subject
        else:
            return image, gaze


class Gaze360(ConcatDataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Callable = None,
        mode: str = "train",
        with_subject: bool = False,
    ):
        self.data_dir = data_dir / "Gaze360.h5"
        self.transform = transform
        self.mode = mode

        if mode == "train":
            mode = [mode]
        elif mode == "val":
            mode = ["test"]  # not use 'val' fold
        elif mode == "all":
            mode = ["train", "test"]
        elif isinstance(mode, str):
            mode = [mode]
        self.subject_datasets = [
            OneFoldGaze360(self.data_dir, self.transform, sub, with_subject)
            for sub in mode
        ]
        super().__init__(self.subject_datasets)
