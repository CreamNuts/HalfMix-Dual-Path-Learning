from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer, lr_scheduler
from torchmetrics import MeanMetric, MinMetric

from .losses import ScalarTargetBCE, ScalarTargetCE, weight_ce, weight_cs
from .metrics import AngularError
from .utils import to_scalar


class GazeLitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        compile: bool = False,
        loss_type="l1",
        num_bins: int = 1,
        num_chunks: int = 2,
        single_mix: bool = False,
        double_head: bool = False,
        weight_ce: bool = False,
        weight_cs: bool = False,
        use_dpr: bool = False,
        use_dgfa: bool = False,
        beta_dgfa: float = 1.0,
        w_cs: float = 0.01,
        w_kl: float = 1.0,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")
        self.net = net
        assert (
            (loss_type == "l1")
            and (num_bins == 1)
            or (loss_type in ["ce", "bce"])
            and (num_bins > 1)
        ), "loss_type and num_bins must match"

        # if double_head:
        #     self.net.set_head(num_chunks * num_bins, DoubleHead)
        # else:
        #     self.net.set_head(num_chunks * num_bins)

        self.naive = not (double_head or single_mix)

        # loss function
        if loss_type == "l1":
            self.lb_type = "scalar"
            self.criterion = nn.L1Loss()
        elif loss_type == "ce":
            self.lb_type = "onehot"
            self.criterion = ScalarTargetCE(num_bins=num_bins, num_chunks=num_chunks)
        elif loss_type == "bce":
            self.lb_type = "ordinal"
            self.criterion = ScalarTargetBCE(num_bins=num_bins, num_chunks=num_chunks)

        # metrics
        self.train_angular = AngularError(lb_type=self.lb_type, num_chunks=num_chunks)
        self.train_loss = MeanMetric()

        self.val_loader_names = ["Source", "MPII", "Eyediap"]  # 실제 순서에 맞게 수정
        self.val_angulars = nn.ModuleList(
            [
                AngularError(lb_type=self.lb_type, num_chunks=num_chunks)
                for _ in range(len(self.val_loader_names))
            ]
        )
        self.val_loss = MeanMetric()
        self.val_angular_best = (
            MinMetric()
        )  # for tracking best so far validation accuracy
        if self.hparams.double_head:
            self.train_ce = MeanMetric()
            self.train_cs = MeanMetric()
            self.train_angular1 = AngularError(
                lb_type=self.lb_type, num_chunks=num_chunks
            )
            self.train_angular2 = AngularError(
                lb_type=self.lb_type, num_chunks=num_chunks
            )
            self.val_angulars1 = nn.ModuleList(
                [
                    AngularError(lb_type=self.lb_type, num_chunks=num_chunks)
                    for _ in range(len(self.val_loader_names))
                ]
            )
            self.val_angulars2 = nn.ModuleList(
                [
                    AngularError(lb_type=self.lb_type, num_chunks=num_chunks)
                    for _ in range(len(self.val_loader_names))
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_angular_best.reset()
        for angular in self.val_angulars:
            angular.reset()
        if self.hparams.double_head:
            for angular1, angular2 in zip(self.val_angulars1, self.val_angulars2):
                angular1.reset()
                angular2.reset()

    def on_train_batch_start(self, batch, batch_idx):
        old_batch = batch
        lam = torch.tensor([0.5], device=batch[0].device)
        if not self.naive:
            old_batch[1] = (batch[1], batch[1])
        old_batch.append(lam)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        old_batch = batch
        lam = torch.tensor([0.5], device=batch[0].device)
        if not self.naive:
            old_batch[1] = (batch[1], batch[1])
        old_batch.append(lam)

    def model_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        x, y, lam = batch
        output = self.forward(x)
        if self.naive:
            target = y
            loss = self.criterion(output, y)
            preds = output
            return loss, preds, target
        elif self.hparams.double_head:
            output1, output2 = output
        elif self.hparams.single_mix:
            output1 = output
            output2 = output

        target = lam * y[0] + (1 - lam) * y[1]
        loss1 = self.criterion(output1, y[0])
        loss2 = self.criterion(output2, y[1])
        loss = lam * loss1 + (1 - lam) * loss2

        if self.lb_type != "scalar":
            pred1 = to_scalar(output1, mode=self.lb_type)
            pred2 = to_scalar(output2, mode=self.lb_type)
            preds = lam * pred1 + (1 - lam) * pred2
        else:
            pred1, pred2 = output1, output2
            preds = lam * output1 + (1 - lam) * output2
        if self.hparams.double_head:
            return loss, (preds, pred1, pred2), (target, y[0], y[1])
        return loss, preds, target

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if self.hparams.double_head:
            loss, (preds, pred1, pred2), (targets, target1, target2) = self.model_step(
                batch
            )
        else:
            loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_angular(preds, targets)
        metrics = {
            "train/loss": self.train_loss,
            "train/angular": self.train_angular,
        }
        # update and log metrics
        if self.hparams.double_head:
            head_weight = self.net.head.head.weight
            if self.hparams.weight_ce:
                ce = weight_ce(head_weight)
                loss = loss + ce
            else:
                with torch.no_grad():
                    ce = weight_ce(head_weight)
            if self.hparams.weight_cs:
                cs = weight_cs(head_weight)
                loss = loss + cs * 0.01
            else:
                with torch.no_grad():
                    cs = weight_cs(head_weight)
            self.train_ce(ce)
            self.train_cs(cs)
            self.train_angular1(pred1, target1)
            self.train_angular2(pred2, target2)
            metrics.update(
                {
                    "train/ce": self.train_ce,
                    "train/cs": self.train_cs,
                    "train/angular1": self.train_angular1,
                    "train/angular2": self.train_angular2,
                }
            )
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.hparams.double_head:
            loss, (preds, pred1, pred2), (targets, target1, target2) = self.model_step(
                batch
            )
        else:
            loss, preds, targets = self.model_step(batch)

        self.val_angulars[dataloader_idx](preds, targets)
        loader_name = self.val_loader_names[dataloader_idx]
        metrics = {
            f"val/angular/{loader_name}": self.val_angulars[dataloader_idx],
        }
        if self.hparams.double_head:
            self.val_angulars1[dataloader_idx](pred1, target1)
            self.val_angulars2[dataloader_idx](pred2, target2)
            metrics.update(
                {
                    f"val/angular1/{loader_name}": self.val_angulars1[dataloader_idx],
                    f"val/angular2{loader_name}": self.val_angulars2[dataloader_idx],
                }
            )
        if dataloader_idx == 0:
            # update and log metrics
            self.val_loss(loss)
            metrics.update({"val/loss": self.val_loss})
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        angular = self.val_angulars[0].compute()  # get current val acc
        self.val_angular_best(angular)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/angular_best",
            self.val_angular_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        # params = add_weight_decay(self.parameters(), weight_decay=self.hparams.weight_decay)
        # optimizer = self.hparams.optimizer(params=params)
        optimizer = self.hparams.optimizer(model=self.net)  # Check create_optimizer
        # optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = GazeLitModule(None, None, None)
