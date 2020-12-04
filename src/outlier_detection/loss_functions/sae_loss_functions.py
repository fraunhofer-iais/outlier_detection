import torch
from ml_gym.loss_functions.loss_scaler import MeanScaler
from ml_gym.batch import InferenceResultBatch
from enum import Enum
from ml_gym.loss_functions.multi_term_loss_functions import MultiLoss
from outlier_detection.loss_functions.ata_loss_functions import AdvReconstructionBinnedLoss, AdvReconstructionThresholdedLoss
from ml_gym.loss_functions.loss_functions import BCEWithLogitsLoss


class SAELoss(MultiLoss):
    class AdversarialType(Enum):
        BINNED = "binned"
        THRESHOLDED = "thresholded"

    def __init__(self,
                 target_class_subscription_key: str,
                 target_original_sample_subscription_key: str,
                 prediction_raw_classification_output_subscription_key: str,
                 prediction_reconstruction_subscription_key: str,
                 outlier_weighting_factor: float = -1,
                 adv_threshold: float = 10,
                 outlier_label: int = 1,
                 loss_lambda=0.5,
                 type: str = "BINNED",
                 tag: str = None
                 ):
        if SAELoss.AdversarialType[type] == SAELoss.AdversarialType.BINNED:
            rec_loss_fun = AdvReconstructionBinnedLoss(target_original_sample_subscription_key=target_original_sample_subscription_key,
                                                       target_class_subscription_key=target_class_subscription_key,
                                                       prediction_reconstruction_subscription_key=prediction_reconstruction_subscription_key,
                                                       outlier_weighting_factor=outlier_weighting_factor,
                                                       outlier_bin_start=adv_threshold,
                                                       outlier_label=outlier_label,
                                                       tag="sae_adv_binned_loss")
        else:
            rec_loss_fun = AdvReconstructionThresholdedLoss(target_original_sample_subscription_key=target_original_sample_subscription_key,
                                                            target_class_subscription_key=target_class_subscription_key,
                                                            prediction_reconstruction_subscription_key=prediction_reconstruction_subscription_key,
                                                            outlier_weighting_factor=outlier_weighting_factor,
                                                            max_adv_threshold=adv_threshold,
                                                            outlier_label=outlier_label,
                                                            tag="sae_adv_thresholded_loss")

        classification_loss_fun = BCEWithLogitsLoss(target_subscription_key=target_class_subscription_key,
                                                    prediction_subscription_key=prediction_raw_classification_output_subscription_key,
                                                    tag="sae_bce_loss")
        loss_terms = [rec_loss_fun, classification_loss_fun]
        scalers = [MeanScaler(), MeanScaler()]
        loss_weights = [1 - loss_lambda, loss_lambda]  # reconstruction loss, classification loss
        MultiLoss.__init__(self, tag, scalers, loss_terms, loss_weights)

    def __call__(self, eval_batch: InferenceResultBatch) -> torch.Tensor:
        losses = MultiLoss.__call__(self, eval_batch)
        return losses
