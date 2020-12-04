from ml_gym.models.nn.net import NNModel
import torch
from torch import nn
from typing import Tuple, Dict, List


class AutoEncoder(NNModel):

    def __init__(self, reconstruction_publication_key: str, encoding_publication_key: str, n_inputs: int, encoder_layer_config: List[int],
                 n_outputs: int, seed: int = 1):
        """
        Instantiates an auto-encoder model

        Arguments:
            n_inputs {int} -- number of inputs
            encoder_layer_config {List[int]} -- encoder layer/decoder layer config excluding the input size

            Note: since the layer sizes are symmetrical for the auto encoder,
            we can specify only one half, in this case the encoder by a list of layer sizes e.g. [100, 50 25]

            n_outputs {int} -- number of outputs
            seed {int} -- seed
        """
        super().__init__(seed=seed)
        self.reconstruction_publication_key = reconstruction_publication_key
        self.encoding_publication_key = encoding_publication_key
        self.encoder, self.decoder = AutoEncoder._get_layers(n_inputs, encoder_layer_config, n_outputs)

    @staticmethod
    def _get_layers(n_inputs: int,
                    encoder_layer_config: List[int],
                    n_outputs: List[int]) -> Tuple[nn.ModuleList, nn.ModuleList]:
        encoder = nn.ModuleList([])
        decoder = nn.ModuleList([])

        # build auto encoder layers
        encoder_layer_config.insert(0, n_inputs)
        for layer_id, input_size in enumerate(encoder_layer_config[:-1]):
            output_size = encoder_layer_config[layer_id + 1]
            # layer = Linear(input_size, output_size, bias=True)
            encoder_layer = nn.Linear(input_size, output_size, bias=True)
            decoder_layer = nn.Linear(output_size, input_size, bias=True)
            encoder.append(encoder_layer)
            decoder.insert(index=0, module=decoder_layer)
        return encoder, decoder

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_impl(inputs)

    def forward_impl(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = inputs
        for encoder_layer in self.encoder:
            output = torch.sigmoid(encoder_layer.forward(output))
        encoded = output

        for i, decoder_layer in enumerate(self.decoder):
            output = decoder_layer.forward(output)
            if (i + 1) != len(self.decoder):  # for the final layer, it is just linear layer, else sigmoidal
                output = torch.sigmoid(output)
        reconstructed = output
        return {self.reconstruction_publication_key: reconstructed,
                self.encoding_publication_key: encoded}
