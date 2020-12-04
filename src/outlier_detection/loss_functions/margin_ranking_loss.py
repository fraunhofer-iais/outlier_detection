import torch
from ml_gym.loss_functions.loss_functions import Loss, LPLoss
from ml_gym.batch import InferenceResultBatch
from enum import Enum
import torch.nn as nn


class MarginRankingLoss(Loss):
    def __init__(self, target_partition: Enum, prediction_partition: Enum, margin: int = 0,
                 first_higher_ranked: int = 1):
        self.target_partition: Enum = target_partition
        self.prediction_partition: Enum = prediction_partition
        self.margin_ranking_loss = nn.MarginRankingLoss(reduction="none", margin=margin)
        self.first_higher_ranked = first_higher_ranked
        self.lp_loss = LPLoss(target_partition, prediction_partition)

    def __call__(self, eval_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Calculates hinge loss
        :return: Loss tensor
        """
        targets = eval_batch.tags
        lp_loss_scores = self.lp_loss(eval_batch)
        higher_ranked_mask = targets == self.first_higher_ranked
        higher_scores = lp_loss_scores[higher_ranked_mask]
        lower_scores = lp_loss_scores[~higher_ranked_mask]
        grid_higher, grid_lower = torch.meshgrid(higher_scores, lower_scores)
        grid_higher_flat, grid_lower_flat = grid_higher.flatten(), grid_lower.flatten()
        y = lp_loss_scores.new_ones(len(grid_higher_flat))
        loss = self.margin_ranking_loss(grid_higher_flat, grid_lower_flat, y)
        return loss