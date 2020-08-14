import torch

class Placeholder(torch.nn.Module):
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
