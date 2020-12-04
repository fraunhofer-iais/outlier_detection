from ml_gym.models.nn.net import NNModel
import torch
from outlier_detection.models.auto_encoder import AutoEncoder
from outlier_detection.models.mlp.model import MLPClassifier
from enum import Enum
from typing import Dict, List, Tuple


class SupervisedAutoEncoder(NNModel):
    class Mode(Enum):
        SUPERVISED_AUTO_ENCODER = "supervised_auto_encoder"
        SUPERVISED_AUTO_ENCODER_WITH_LOSS_READOUT = "supervised_auto_encoder_with_loss_readout"
        SUPERVISED_AUTO_ENCODER_ONLY_LOSS_READOUT = "supervised_auto_encoder_only_loss_readout"

    def __init__(self,
                 reconstruction_publication_key: str,
                 encoding_publication_key: str,
                 raw_classification_output_publication_key: str,
                 lp_loss_publication_key: str,
                 mode: str,
                 n_inputs: int,
                 auto_encoder_layer_config: List[int],
                 read_out_hidden_layer_config: List[int],
                 n_outputs: int = 1,
                 seed: int = 1):
        super().__init__(seed=seed)
        self.reconstruction_publication_key = reconstruction_publication_key
        self.encoding_publication_key = encoding_publication_key
        self.raw_classification_output_publication_key = raw_classification_output_publication_key
        self.lp_loss_publication_key = lp_loss_publication_key

        self.mode = SupervisedAutoEncoder.Mode[mode]
        # build auto encoder and read out modules
        self.auto_encoder_module, self.read_out_module = SupervisedAutoEncoder._build_modules(reconstruction_publication_key,
                                                                                              encoding_publication_key,
                                                                                              raw_classification_output_publication_key,
                                                                                              self.mode,
                                                                                              n_inputs,
                                                                                              auto_encoder_layer_config,
                                                                                              read_out_hidden_layer_config,
                                                                                              n_outputs,
                                                                                              seed)

    @staticmethod
    def _build_modules(reconstruction_publication_key: str,
                       encoding_publication_key: str,
                       raw_classification_output_publication_key: str,
                       mode: 'SupervisedAutoEncoder.Mode',
                       n_inputs: int,
                       auto_encoder_layer_config: List[int],
                       read_out_hidden_layer_config: List[int],
                       n_outputs: int,
                       seed: int) -> Tuple[AutoEncoder, MLPClassifier]:
        auto_encoder = AutoEncoder(reconstruction_publication_key, encoding_publication_key,
                                   n_inputs, auto_encoder_layer_config, n_inputs, seed)

        if mode == SupervisedAutoEncoder.Mode.SUPERVISED_AUTO_ENCODER_WITH_LOSS_READOUT:
            read_out_input_size = 1 + auto_encoder_layer_config[-1]
        elif mode == SupervisedAutoEncoder.Mode.SUPERVISED_AUTO_ENCODER:
            read_out_input_size = auto_encoder_layer_config[-1]
        else:  # SUPERVISED_AUTO_ENCODER_ONLY_LOSS_READOUT
            read_out_input_size = 1
        read_out = MLPClassifier(raw_classification_output_publication_key, read_out_input_size, read_out_hidden_layer_config, n_outputs, seed)
        return auto_encoder, read_out

    @classmethod
    def read_out_lp_loss(cls, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        exponent = 2
        root = 1
        read_out_loss = (torch.sum((outputs - inputs).abs() ** exponent, dim=1) ** (1 / root)) / inputs.shape[1]
        return read_out_loss

    def forward(self, inputs: torch.Tensor) -> Dict[Enum, torch.Tensor]:
        return self.forward_impl(inputs)

    def forward_impl(self, inputs: torch.Tensor) -> Dict[Enum, torch.Tensor]:
        ae_output_dict = self.auto_encoder_module.forward_impl(inputs)

        # compute the reconstruction loss
        reconstructed = ae_output_dict[self.reconstruction_publication_key]
        reconstruction_loss = self.read_out_lp_loss(inputs=inputs, outputs=reconstructed)

        # classification output from the readout on the reconstruction loss
        output = ae_output_dict[self.encoding_publication_key]
        if self.mode == SupervisedAutoEncoder.Mode.SUPERVISED_AUTO_ENCODER_WITH_LOSS_READOUT:
            output = torch.cat((reconstruction_loss.unsqueeze(dim=1), output), dim=1)
        elif self.mode == SupervisedAutoEncoder.Mode.SUPERVISED_AUTO_ENCODER_ONLY_LOSS_READOUT:
            output = reconstruction_loss.unsqueeze(dim=1)

        output = self.read_out_module(output)
        output = output[self.raw_classification_output_publication_key]
        return {self.raw_classification_output_publication_key: output,
                self.lp_loss_publication_key: reconstruction_loss,
                self.reconstruction_publication_key: reconstructed,
                self.encoding_publication_key: ae_output_dict[self.encoding_publication_key]}
