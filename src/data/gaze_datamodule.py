from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from lightning import LightningDataModule
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    RandomResizedCropAndInterpolation,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)

from src.data.components import EthXGaze, Eyediap, Gaze360, MPIIFaceGaze


def create_transform(
    img_size: Union[int, Tuple[int, int]] = 224,
    train: bool = False,
    scale: Tuple[float, float] = (0.8, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    color_jitter: float = 0.4,
    mean: Union[float, Tuple[float, float, float]] = IMAGENET_DEFAULT_MEAN,
    std: Union[float, Tuple[float, float, float]] = IMAGENET_DEFAULT_STD,
):
    """
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    if train:
        tfl = [
            ToPILImage(),
            RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio),
        ]

        if color_jitter is not None:
            # color jitter is enabled when not using AA
            if isinstance(color_jitter, (list, tuple)):
                # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
                # or 4 if also augmenting hue
                assert len(color_jitter) in (3, 4)
            else:
                # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
                color_jitter = (float(color_jitter),) * 3
            tfl += [ColorJitter(*color_jitter)]
    else:
        tfl = [ToPILImage(), Resize(img_size)]

    final_tfl = [
        ToTensor(),
        Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    transform = Compose(tfl + final_tfl)
    return transform


class GazeDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: str,
        with_subject: bool = False,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        aug: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transforms = create_transform(
            train=aug,
            scale=(0.8, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            color_jitter=0.4,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        self.val_transforms = create_transform(
            train=False,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

        self.data_train: Optional[Dataset] = None
        self.data_vals: List[Optional[Dataset]] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        STR2DATASET = {
            "ethxgaze": partial(EthXGaze, with_subject=self.hparams.with_subject),
            "gaze360": partial(Gaze360, with_subject=self.hparams.with_subject),  # Gaze360 dataset does not support subject, it returns fold number instead.
            "mpii": partial(MPIIFaceGaze, with_subject=self.hparams.with_subject),
            "eyediap": partial(Eyediap, with_subject=self.hparams.with_subject),
        }
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
        if self.hparams.train_dataset == "ethxgaze":
            STR2DATASET.pop("gaze360")
        elif self.hparams.train_dataset == "gaze360":
            STR2DATASET.pop("ethxgaze")
        # load and split datasets only if not loaded already
        data_dir = Path(self.hparams.data_dir)
        if not self.data_train and not self.data_vals:
            self.data_train = STR2DATASET[self.hparams.train_dataset](
                data_dir, transform=self.transforms, mode="train"
            )
            data_vals = []
            for name, dataset in STR2DATASET.items():
                if name == self.hparams.train_dataset:
                    valset = dataset(data_dir, transform=self.transforms, mode="val")
                    data_vals.insert(0, valset)
                else:
                    valset = dataset(data_dir, transform=self.transforms, mode="all")
                    data_vals.append(valset)
            self.data_vals = data_vals

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        loaders = []
        for data_val in self.data_vals:
            loaders.append(
                DataLoader(
                    dataset=data_val,
                    batch_size=self.batch_size_per_device,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                )
            )
        return loaders

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass
