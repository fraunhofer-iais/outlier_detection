from ml_gym.gym.trainer import TrainComponent
from ml_gym.gym.inference_component import InferenceComponent
from typing import List, Callable
import torch
import scipy
from outlier_detection.models.ata.model import AdversariallyTrainedAutoEncoder
from ml_gym.models.nn.net import NNModel
from ml_gym.loss_functions.loss_functions import Loss
from ml_gym.data_handling.dataset_loader import DatasetLoader
from ml_gym.batch import InferenceResultBatch
from torch.optim.optimizer import Optimizer
import numpy as np


class ATATrainComponent(TrainComponent):
    def __init__(self, prediction_lp_loss_subscription_key: str, target_class_subscription_key: str, inference_component: InferenceComponent,
                 loss_fun: Loss, score_fun: Callable[[List[int], List[float]], float]):
        super().__init__(inference_component, loss_fun)
        self.score_fun = score_fun
        self.prediction_lp_loss_subscription_key = prediction_lp_loss_subscription_key
        self.target_class_subscription_key = target_class_subscription_key

    def train_epoch(self, model: AdversariallyTrainedAutoEncoder, optimizer: Optimizer, data_loader: DatasetLoader,
                    device: torch.device) -> NNModel:
        model = super().train_epoch(model, optimizer, data_loader, device)
        threshold = self.tune_threshold(model, data_loader, device)
        model.threshold = threshold
        return model

    def tune_threshold(self, model: AdversariallyTrainedAutoEncoder, data_loader: DatasetLoader, device: torch.device) -> float:
        prediction_batches = self.map_batches(fun=self.forward_batch,
                                              fun_params={"device": device,
                                                          "model": model},
                                              loader=data_loader)
        prediction_batch = InferenceResultBatch.combine(prediction_batches)
        targets = prediction_batch.targets[self.target_class_subscription_key].cpu().numpy()
        read_outs = prediction_batch.predictions[self.prediction_lp_loss_subscription_key].detach().cpu().numpy()
        upper_bound = np.median(read_outs[targets == 1])
        lower_bound = 0
        step_size = float(upper_bound - lower_bound) / 250
        # apply scipy minimize
        result = scipy.optimize.brute(self._compute_score, (slice(lower_bound, upper_bound, step_size),),
                                      args=(read_outs, targets, self.score_fun),
                                      full_output=True,
                                      workers=1)
        threshold = torch.tensor(result[0][0]).float().item()
        return threshold

    def _compute_score(self, threshold, reconstruction_losses: List[float], targets: List[int],
                       score_fun: Callable[[List[int], List[float]], float]):
        # apply thresholding and create binary predictions
        outputs = torch.zeros(len(reconstruction_losses))
        outlier_indices = np.array(reconstruction_losses) >= threshold
        outputs[outlier_indices] = 1  # TODO this is hardcoded!
        outputs = outputs.numpy().tolist()
        score = score_fun(targets, outputs)
        return score
