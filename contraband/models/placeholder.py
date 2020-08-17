import torch


class Placeholder(torch.nn.Module):
    """
        This model is used when doing predictions directly on embeddings.
        It allows the training infrastructure to be used as normal, but
        not pass data through the normal base_encoder.

        Note that the out_shape is None.
    """
    def __init__(self):
        super().__init__()

        self.name = "Placeholder"

    def make_model(self, model_params):
        self.in_shape = model_params['in_shape']
        self.dims = len(self.in_shape)

        self.out_channels = model_params["h_channels"] 

        self.out_shape = None 

    def forward(self, raw):
        return raw
