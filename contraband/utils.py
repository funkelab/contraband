import torch
import logging


def load_model(model, prefix, checkpoint_file):
    """Loads the model from the given checkpoint and prefix.

    Args:

        model (:class:`torch.nn.modual`):

            The model to load.

        prefix (``str``):
            
            This is the prefix of the model in the checkpoint file.
            Ex: Unet2D has unet and when it gets saved it is stored
            as unet.layer0, unet.layer1, ... so you should give it the
            prefix 'unet.'. The period at the end is important or it 
            will not be parsed correctly. 
    """
    checkpoint = torch.load(checkpoint_file)
    loaded_dict = checkpoint['model_state_dict']
    print(loaded_dict.keys())
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() 
                    if k.startswith(prefix)}
    adapted_dict = {k: v for k, v in adapted_dict.items() 
                    if not k.startswith("projection_head")}
    model.load_state_dict(adapted_dict)
    
    return model


def create_logger(log_dir, name=None, index=None):
    if name is None:
        assert index is not None, "Must specify index in create logger"
        name = 'combination-' + str(index)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_dir + '/' + name + ".log")
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
