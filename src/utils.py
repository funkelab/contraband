import torch

def load_model(model, prefix, checkpoint_file):
    """Loads the model from the given checkpoint and prefix.

    Args:

        model (:class:`torch.nn.modual`):

            The model to load.

        prefix (``str``):
            
            This is the prefix of the model in the checkpoint file.
            Ex: Unet2D has self.unet and when it gets saved it is stored
            as unet.layer0, unet.layer1, ... so you should give it the
            prefix 'unet.'. The period at the end is important or it 
            will not be parsed correctly. 
    """
    checkpoint = torch.load(checkpoint_file)
    loaded_dict = checkpoint['model_state_dict']
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() 
                    if k.startswith(prefix)}
    model.load_state_dict(adapted_dict)
    
    return model


