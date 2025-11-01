from pathlib import Path
from typing import Callable

import h5py
import torch
from torch.utils.data import ConcatDataset, Dataset

from .util import get_num
from .var import (
    ETHXGAZE_ALL,
    ETHXGAZE_PART_TRAIN,
    ETHXGAZE_SPLIT,
    ETHXGAZE_TEST,
    ETHXGAZE_TRAIN,
    ETHXGAZE_VAL,
)


class OneSubjectEthX(Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Callable,
        mode: str = "train",
        with_subject: bool = False,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        if mode in ["train", "val", "all", "part"]:
            self.train = True
        elif mode == "test":
            self.train = False
        self.with_subject = with_subject
        if self.with_subject:
            self.subject = get_num(data_dir.name)
        self.hdf5 = h5py.File(self.data_dir, "r", swmr=True)

    def __len__(self) -> int:
        return len(self.hdf5["face_patch"])

    def __getitem__(self, idx: int):
        image = self.hdf5["face_patch"][idx][:, :, ::-1]
        # with h5py.File(self.data_dir, "r", swmr=True) as f:
        #     image = f["face_patch"][idx]
        #     if self.train:
        #         gaze = f["face_gaze"][idx].astype("float")
        if self.transform:
            image = self.transform(image)
        if self.train:
            gaze = self.hdf5["face_gaze"][idx].astype("float")
            gaze = torch.from_numpy(gaze)
            if self.with_subject:
                return image, gaze, self.subject
            else:
                return image, gaze
        return image


class EthXGaze(ConcatDataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Callable,
        mode: bool = "train",
        with_subject: bool = False,
    ):
        self.data_dir = data_dir / "xgaze_224"
        self.transform = transform
        self.mode = mode
        self.split = ETHXGAZE_SPLIT["train"]
        if mode == "train":
            self.sub_dir = "train"
            self.subject_list = ETHXGAZE_TRAIN
        elif mode == "part":
            self.sub_dir = "train"
            self.subject_list = ETHXGAZE_PART_TRAIN
        elif mode == "val":
            self.sub_dir = "train"
            self.subject_list = ETHXGAZE_VAL
        elif mode == "test":
            self.sub_dir = "test"
            self.subject_list = ETHXGAZE_TEST
            self.split = ETHXGAZE_SPLIT["test"]
        elif mode == "all":
            self.sub_dir = "train"
            self.subject_list = ETHXGAZE_ALL
        self.subject_datasets = [
            OneSubjectEthX(
                self.data_dir / self.sub_dir / subject,
                transform,
                mode,
                with_subject,
            )
            for subject in self.split
            if get_num(subject) in self.subject_list
        ]
        super().__init__(self.subject_datasets)
