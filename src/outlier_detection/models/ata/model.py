from ml_gym.models.nn.net import NNModel
import torch
from outlier_detection.models.auto_encoder import AutoEncoder
from typing import List


class AdversariallyTrainedAutoEncoder(NNModel):

    def __init__(self, class_prediction_publication_key: str, lp_loss_publication_key: str, reconstruction_publication_key: str,
                 encoding_publication_key: str, n_inputs: int, encoder_layers: List[int], n_outputs: int, seed=1):
        super().__init__(seed=seed)
        self.class_prediction_publication_key = class_prediction_publication_key
        self.reconstruction_publication_key = reconstruction_publication_key
        self.encoding_publication_key = encoding_publication_key
        self.lp_loss_publication_key = lp_loss_publication_key
        self.n_inputs = n_inputs
        self.encoder_layers = encoder_layers
        # build layers
        self.auto_encoder = AutoEncoder(reconstruction_publication_key=reconstruction_publication_key,
                                        encoding_publication_key=encoding_publication_key,
                                        n_inputs=n_inputs,
                                        encoder_layer_config=encoder_layers,
                                        n_outputs=n_inputs,
                                        seed=seed)

        # threshold gets set from outside via brute force optimization in TrainComponent
        self.threshold = torch.tensor(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_dict = self.forward_impl(inputs)
        read_out = output_dict[self.lp_loss_publication_key]
        prediction = self._map_readout_to_class(read_out)
        output_dict[self.class_prediction_publication_key] = prediction
        return output_dict

    def forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        output_dict = self.auto_encoder.forward_impl(inputs)
        reconstructed = output_dict[self.reconstruction_publication_key]
        # compute the reconstruction loss
        reconstruction_loss = self._read_out_lp_loss(inputs=inputs, outputs=reconstructed)
        output_dict[self.lp_loss_publication_key] = reconstruction_loss
        return output_dict

    @staticmethod
    def _read_out_lp_loss(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        exponent = 2
        root = 1
        read_out_loss = (torch.sum((outputs - inputs).abs() ** exponent, dim=1) ** (1 / root)) / inputs.shape[1]
        return read_out_loss

    def _map_readout_to_class(self, read_out: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros(read_out.shape[0]).int()
        # apply thresholding and create binary predictions
        outlier_indices = read_out >= self.threshold
        outputs[outlier_indices] = 1
        return outputs
