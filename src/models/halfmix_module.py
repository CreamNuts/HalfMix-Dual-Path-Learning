import numpy as np
import torch
from torchmetrics import MeanMetric

from .gaze_module import GazeLitModule


def halfmix(image, target, prob_thresh=0.5):
    """HalfMix augmentation as described in the paper.
    Combines halves of two images while preserving eye regions."""
    prob = np.random.rand()
    if prob > prob_thresh:
        batch_size = image.size(0)
        H = image.size(2)
        W = image.size(3)
        index = torch.randperm(batch_size)

        # Combine left half of original with right half of shuffled images
        mixed_image = image.clone()
        mixed_image[:, :, :, W // 2 :] = image[index, :, :, W // 2 :]

        lam = torch.tensor([0.5], device=image.device)

        # target1: left (original), target2: right (shuffled)
        target1 = target
        target2 = target[index]
    else:
        mixed_image = image
        target1 = target
        target2 = target
        lam = torch.tensor([0.5], device=image.device)
    return mixed_image, (target1, target2), lam


class HalfMixLitModule(GazeLitModule):
    def __init__(
        self,
        num_chunks: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            num_chunks=num_chunks,
            **kwargs,
        )

        # Initialize metrics for regularization losses
        self.train_dgfa_loss = MeanMetric()
        self.train_dpr_loss = MeanMetric()

    def on_train_batch_start(self, batch, batch_idx):
        old_batch = batch
        image, target = batch

        # halfmix 함수 호출
        image, (target1, target2), lam = halfmix(image, target, prob_thresh=0.5)

        old_batch[0] = image
        old_batch[1] = (target1, target2)
        old_batch.append(lam)

    def dual_gaze_feature_alignment(
        self, feature1, feature2, target1, target2, temperature=0.07
    ):
        """Dual-Gaze Feature Alignment (DGFA) from the paper.
        Encourages similar features for similar gaze directions."""
        # Normalize features
        z1 = torch.nn.functional.normalize(feature1, dim=1)
        z2 = torch.nn.functional.normalize(feature2, dim=1)

        # Gaze similarity based on angular distance
        gaze_distance = torch.norm(target1 - target2, dim=1).float()
        s_gaze = torch.exp(-gaze_distance / temperature)

        # Feature similarity
        s_feat = torch.nn.functional.cosine_similarity(z1, z2, dim=1)

        # DGFA loss
        loss = torch.nn.functional.mse_loss(s_feat, s_gaze)
        return loss

    def diversity_promoting_regularization(self, weight1, weight2, w_cs=0.01, w_kl=1.0):
        """Diversity-Promoting Regularization (DPR) from the paper.
        Prevents redundant feature learning between dual paths."""
        # Cosine similarity penalty
        w1_norm = torch.nn.functional.normalize(weight1.view(-1), dim=0)
        w2_norm = torch.nn.functional.normalize(weight2.view(-1), dim=0)
        l_cs = torch.dot(w1_norm, w2_norm)

        if w_kl > 0.0:
            # Convert weights to probability distributions
            q1 = torch.nn.functional.softmax(weight1.view(-1).abs(), dim=0)
            q2 = torch.nn.functional.softmax(weight2.view(-1).abs(), dim=0)

            # Symmetric KL divergence
            kl_1_2 = torch.nn.functional.kl_div(q1.log(), q2, reduction="batchmean")
            kl_2_1 = torch.nn.functional.kl_div(q2.log(), q1, reduction="batchmean")
            l_kl = 0.5 * (kl_1_2 + kl_2_1)
        else:
            l_kl = 0.0
        # Total DPR loss
        l_dpr = w_cs * l_cs + w_kl * l_kl
        return l_dpr, l_cs, l_kl

    def training_step(self, batch, batch_idx):
        # Get batch data
        x, y, lam = batch
        target1, target2 = y

        # Forward pass
        output = self.forward(x)
        output1, output2, feature1, feature2, common_features = output

        # Calculate supervised losses
        loss1 = self.criterion(output1, target1)
        loss2 = self.criterion(output2, target2)

        # Dual-Gaze Feature Alignment (DGFA)
        dgfa_loss = self.dual_gaze_feature_alignment(
            feature1, feature2, target1, target2
        )

        # Total loss calculation with hyperparameters from paper
        loss = lam * loss1 + (1 - lam) * loss2

        # Apply DGFA loss
        if self.hparams.get("use_dgfa", True):
            beta = self.hparams.get("beta_dgfa", 1.0)  # β in paper
            loss = loss + beta * dgfa_loss

        # Diversity-Promoting Regularization (DPR) - backward 전에 계산
        if self.hparams.double_head:
            proj1_weight = self.net.head.proj1.weight  # (in_features, in_features)
            proj2_weight = self.net.head.proj2.weight  # (in_features, in_features)
            head1_weight = self.net.head.head1.weight  # (out_features, in_features)
            head2_weight = self.net.head.head2.weight  # (out_features, in_features)

            proj_dpr_loss, proj_l_cs, proj_l_kl = (
                self.diversity_promoting_regularization(
                    proj1_weight,
                    proj2_weight,
                    w_cs=self.hparams.get("w_cs", 0.01),
                    w_kl=0.0,
                )
            )

            dpr_loss, l_cs, l_kl = self.diversity_promoting_regularization(
                head1_weight,
                head2_weight,
                w_cs=self.hparams.get("w_cs", 0.01),
                w_kl=self.hparams.get("w_kl", 1.0),
            )

            # Apply DPR loss
            if self.hparams.get("use_dpr", True):
                loss = loss + dpr_loss + proj_dpr_loss

        with torch.no_grad():
            # Calculate predictions for metrics
            pred1, pred2 = output1, output2
            preds = lam * output1 + (1 - lam) * output2
            target = lam * target1 + (1 - lam) * target2

            # Update metrics
            self.train_loss(loss)
            self.train_angular(preds, target)
            self.train_angular1(pred1, target1)
            self.train_angular2(pred2, target2)
            self.train_dgfa_loss(dgfa_loss)

            # Log metrics
            self.log(
                "train/loss",
                self.train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("train/angular", self.train_angular, on_step=False, on_epoch=True)
            self.log(
                "train/angular1", self.train_angular1, on_step=False, on_epoch=True
            )
            self.log(
                "train/angular2", self.train_angular2, on_step=False, on_epoch=True
            )
            self.log("train/dgfa", self.train_dgfa_loss, on_step=False, on_epoch=True)

            # Log DPR metrics if applicable
            if self.hparams.double_head and self.hparams.get("use_dpr", True):
                self.log("train/dpr", dpr_loss, on_step=False, on_epoch=True)
                self.log("train/dpr_cs", l_cs, on_step=False, on_epoch=True)
                self.log("train/dpr_kl", l_kl, on_step=False, on_epoch=True)
                self.log("train/proj_dpr", proj_dpr_loss, on_step=False, on_epoch=True)
                self.log("train/proj_dpr_cs", proj_l_cs, on_step=False, on_epoch=True)
                self.log("train/proj_dpr_kl", proj_l_kl, on_step=False, on_epoch=True)
        return loss


if __name__ == "__main__":
    print("HalfMixLitModule with DPR and DGFA loaded successfully.")
