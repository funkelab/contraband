import os
import torch
import logging
import numpy as np

def load_model(model, prefix, checkpoint_file, freeze_model=False):
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
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False
    
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


def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape


def get_checkpoints(path, match=None):
    print(path)
    if match is None:
        checkpionts = [filename for filename in os.listdir(path)]
    else:
        checkpionts = [filename for filename in os.listdir(path) if match in filename]
    checkpionts.sort()
    print(checkpionts)

    return checkpionts


def get_history(path):
    if os.path.isfile(path):
        history = np.load(path)
        start_idx = history.shape[0]
        return history.tolist(), start_idx
    else:
        return [], 0
