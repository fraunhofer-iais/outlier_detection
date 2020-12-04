import torch
from ml_gym.loss_functions.loss_functions import LPLoss, LossWarmupMixin, Loss
from ml_gym.loss_functions.loss_scaler import MeanScaler
from ml_gym.batch import InferenceResultBatch


class AdvReconstructionBinnedLoss(LossWarmupMixin, Loss):

    def __init__(self, target_original_sample_subscription_key: str, target_class_subscription_key: str,
                 prediction_reconstruction_subscription_key: str, root: int = 1, exponent: int = 2,
                 outlier_weighting_factor: float = -1, outlier_bin_start: float = 10, outlier_label: int = 1, tag: str = None):
        LossWarmupMixin.__init__(self)
        Loss.__init__(self, tag)

        #
        # | inliers ... | outliers | ...
        # 0             u        u*1.1
        # The idea is to move the inlier's loss as close to 0 as possible, while the outliers are
        # captured within the interval [u, u*1.1]. This means that we always minimize inliers, we maximize outliers in
        # the range [0, u] and minimize it within the interval [u*1.1, inf]. We do not optimize outliers that already
        # fall into the interval [u, u*1.1]
        self.inlier_scaler = MeanScaler()
        self.outlier_scaler = MeanScaler()
        self.lp_loss_fun = LPLoss(target_subscription_key=target_original_sample_subscription_key,
                                  prediction_subscription_key=prediction_reconstruction_subscription_key,
                                  root=root,
                                  exponent=exponent,
                                  sample_selection_fun=None,
                                  tag=tag)
        self.outlier_weighting_factor = outlier_weighting_factor
        self.outlier_bin = [outlier_bin_start, outlier_bin_start * 1.1]
        self.outlier_label = outlier_label
        self.warmup_losses = {"outlier": [], "inlier": []}
        self.target_class_subscription_key = target_class_subscription_key

    def __call__(self, eval_batch: InferenceResultBatch) -> torch.Tensor:
        loss_tensor = self._calc_loss(eval_batch)
        return loss_tensor

    def warm_up(self, forward_batch: InferenceResultBatch):
        outliers_mask = forward_batch.get_targets(key=self.target_class_subscription_key) == self.outlier_label
        # calculate losses
        loss_tensor = self.lp_loss_fun(forward_batch)
        loss_tensor_outliers = loss_tensor[outliers_mask]
        loss_tensor_inliers = loss_tensor[~outliers_mask]
        self.warmup_losses["outlier"].append(loss_tensor_outliers)
        self.warmup_losses["inlier"].append(loss_tensor_inliers)
        return loss_tensor

    def finish_warmup(self):
        inlier_loss_tensor = torch.cat(self.warmup_losses["inlier"])
        outlier_loss_tensor = torch.cat(self.warmup_losses["outlier"])
        self.inlier_scaler.train(inlier_loss_tensor)
        self.outlier_scaler.train(outlier_loss_tensor)

    def _calc_loss(self, eval_batch: InferenceResultBatch) -> torch.Tensor:
        outliers_mask = eval_batch.get_targets(key=self.target_class_subscription_key) == self.outlier_label

        # calculate losses
        loss_tensor = self.lp_loss_fun(eval_batch)

        # scale losses
        loss_tensor[outliers_mask] = self.outlier_scaler.scale(loss_tensor[outliers_mask])
        loss_tensor[~outliers_mask] = self.inlier_scaler.scale(loss_tensor[~outliers_mask])

        # nullify outliers within bin
        loss_tensor_outliers = loss_tensor[outliers_mask]
        nullify_mask = ~torch.mul(loss_tensor_outliers > self.outlier_bin[0],
                                  loss_tensor_outliers < self.outlier_bin[1]).bool()
        loss_tensor_outliers = loss_tensor_outliers * nullify_mask
        # maximize left side of the bin
        loss_tensor_outliers[loss_tensor_outliers < self.outlier_bin[0]] = loss_tensor_outliers[
            loss_tensor_outliers < self.outlier_bin[0]] * self.outlier_weighting_factor

        loss_tensor[outliers_mask] = loss_tensor_outliers
        return loss_tensor


class AdvReconstructionThresholdedLoss(LossWarmupMixin, Loss):
    # TODO we have ugly code duplication with AdvReconstructionBinnedLoss here!!!

    def __init__(self, target_original_sample_subscription_key: str, target_class_subscription_key: str,
                 prediction_reconstruction_subscription_key: str, root: int = 1, exponent: int = 2,
                 outlier_weighting_factor: float = -1, max_adv_threshold: float = 10, outlier_label: int = 1, tag: str = None):
        LossWarmupMixin.__init__(self)
        Loss.__init__(self, tag)

        #
        # | inliers ... | outliers ...
        # 0             t
        # The idea is to move the inlier's loss as close to 0 as possible, while the outliers are
        # pushed over threshold t. This means that we always minimize inliers, we maximize outliers in
        # the range [0, t] and don't touch them once they are within [t, inf]
        self.inlier_scaler = MeanScaler()
        self.outlier_scaler = MeanScaler()
        self.lp_loss_fun = LPLoss(target_subscription_key=target_original_sample_subscription_key,
                                  prediction_subscription_key=prediction_reconstruction_subscription_key,
                                  root=root,
                                  exponent=exponent,
                                  sample_selection_fun=None,
                                  tag=tag)
        self.outlier_weighting_factor = outlier_weighting_factor
        self.max_adv_threshold = max_adv_threshold
        self.outlier_label = outlier_label
        self.warmup_losses = {"outlier": [], "inlier": []}
        self.target_class_subscription_key = target_class_subscription_key

    def __call__(self, eval_batch: InferenceResultBatch) -> torch.Tensor:
        loss_tensor = self._calc_loss(eval_batch)
        return loss_tensor

    def warm_up(self, forward_batch: InferenceResultBatch):
        outliers_mask = forward_batch.get_targets(key=self.target_class_subscription_key) == self.outlier_label
        # calculate losses
        loss_tensor = self.lp_loss_fun(forward_batch)
        loss_tensor_outliers = loss_tensor[outliers_mask]
        loss_tensor_inliers = loss_tensor[~outliers_mask]
        self.warmup_losses["outlier"].append(loss_tensor_outliers)
        self.warmup_losses["inlier"].append(loss_tensor_inliers)
        return loss_tensor

    def finish_warmup(self):
        inlier_loss_tensor = torch.cat(self.warmup_losses["inlier"])
        outlier_loss_tensor = torch.cat(self.warmup_losses["outlier"])
        self.inlier_scaler.train(inlier_loss_tensor)
        self.outlier_scaler.train(outlier_loss_tensor)

    def _calc_loss(self, eval_batch: InferenceResultBatch) -> torch.Tensor:
        outliers_mask = eval_batch.get_targets(key=self.target_class_subscription_key) == self.outlier_label
        # calculate losses
        loss_tensor = self.lp_loss_fun(eval_batch)
        # scale losses
        loss_tensor[outliers_mask] = self.outlier_scaler.scale(loss_tensor[outliers_mask])
        loss_tensor[~outliers_mask] = self.inlier_scaler.scale(loss_tensor[~outliers_mask])
        # nullify outliers within bin
        loss_tensor_outliers = loss_tensor[outliers_mask]
        nullify_mask = (loss_tensor_outliers < self.max_adv_threshold).bool()
        loss_tensor_outliers = loss_tensor_outliers * nullify_mask
        # maximize left side of the bin
        loss_tensor_outliers[loss_tensor_outliers < self.max_adv_threshold] = loss_tensor_outliers[
            loss_tensor_outliers < self.max_adv_threshold] * self.outlier_weighting_factor

        loss_tensor[outliers_mask] = loss_tensor_outliers
        return loss_tensor
