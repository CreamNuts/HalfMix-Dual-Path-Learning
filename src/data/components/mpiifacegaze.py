from pathlib import Path
from typing import Callable, Union

import h5py
import torch
from torch.utils.data import ConcatDataset, Dataset

from .util import get_num
from .var import MPIIFACE_FOLDS


class OneSubjectMPII(Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Callable,
        mode: str,
        with_subject: bool = False,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.with_subject = with_subject
        if self.with_subject:
            self.subject = get_num(self.mode)

    def __len__(self):
        with h5py.File(self.data_dir, "r", swmr=True) as f:
            return len(f[self.mode]["face_patch"])

    def __getitem__(self, idx):
        with h5py.File(self.data_dir, "r", swmr=True) as f:
            image = f[self.mode]["face_patch"][idx]
            gaze = f[self.mode]["face_gaze"][idx].astype("float")
        image = self.transform(image)
        gaze = torch.from_numpy(gaze)
        if self.with_subject:
            return image, gaze, self.subject
        else:
            return image, gaze


class MPIIFaceGaze(ConcatDataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Callable,
        mode: Union[str, list[str]],
        with_subject: bool = False,
    ):
        self.data_dir = data_dir / "MPIIFaceGaze.h5"
        self.transform = transform
        self.mode = mode
        if mode == "train":
            mode = MPIIFACE_FOLDS[5:]
        elif mode == "val":
            mode = MPIIFACE_FOLDS[:5]
        elif mode == "all":
            mode = MPIIFACE_FOLDS
        elif isinstance(mode, str):
            mode = [mode]
        self.subject_datasets = [
            OneSubjectMPII(self.data_dir, self.transform, sub, with_subject)
            for sub in mode
        ]
        super().__init__(self.subject_datasets)
