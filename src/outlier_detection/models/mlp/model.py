from ml_gym.models.nn.net import NNModel
from typing import List, Dict
from torch import nn
import torch


class MLPClassifier(NNModel):
    def __init__(self, prediction_publication_key: str, n_inputs: int, hidden_layer_config: List[int], n_outputs: int, seed=0):
        super(MLPClassifier, self).__init__(seed=seed)
        self.layers = MLPClassifier._get_layers(n_inputs, n_outputs, hidden_layer_config)
        self.prediction_publication_key = prediction_publication_key

    def _partition_target_tensor(self, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {self.label_partition: targets}

    @staticmethod
    def _get_layers(n_inputs: int, n_outputs: int, hidden_layer_config: List[int]) -> nn.ModuleList:
        layers = nn.ModuleList()
        input_size = n_inputs
        # input to hidden, hidden to hidden
        for layer_id, hidden_size in enumerate(hidden_layer_config):
            layer = nn.Linear(input_size, hidden_size)
            input_size = hidden_size
            layers.append(layer)
        # output layer
        layers.append(nn.Linear(input_size, n_outputs))

        return layers

    def forward_impl(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pass the input to all layers
        output = inputs
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # NOTE: the last layer does not have a sigmoid! This is what we need e.g., for BCELossWithLogits.
            output = torch.sigmoid(layer(output)) if i != len(self.layers) - 1 else layer(output)
        return {self.prediction_publication_key: output}

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_impl(inputs)
