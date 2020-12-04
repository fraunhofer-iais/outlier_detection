from ml_gym.gym.evaluator import EvalComponent
from ml_gym.gym.inference_component import InferenceComponent
from typing import List, Dict
import torch
from outlier_detection.models.ata.model import AdversariallyTrainedAutoEncoder
from ml_gym.loss_functions.loss_functions import Loss
from ml_gym.data_handling.dataset_loader import DatasetLoader
from ml_gym.batch import EvaluationBatchResult
from ml_gym.metrics.metrics import Metric


class ATAEvalComponent(EvalComponent):
    """This thing always comes with batteries included, i.e., datasets, loss functions etc. are all already
    stored in here."""

    def __init__(self, inference_component: InferenceComponent, metrics: List[Metric],
                 loss_funs: Dict[str, Loss], dataset_loaders: Dict[str, DatasetLoader], train_split_name: str,
                 average_batch_loss: bool = True):

        super().__init__(inference_component, metrics, loss_funs, dataset_loaders, train_split_name, average_batch_loss)

    def evaluate(self, model: AdversariallyTrainedAutoEncoder, device: torch.device) -> List[EvaluationBatchResult]:
        eval_result_list = super().evaluate(model, device)
        # we set the threshold value in the first split only (usually train)
        eval_result_list[0].metrics["threshold"] = [float(model.threshold)]
        return eval_result_list
