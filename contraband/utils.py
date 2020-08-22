import sys
import os
import torch
import logging
import numpy as np
import daisy


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
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() 
                    if k.startswith(prefix)}
    # We never want to load the projection_head
    # GP will take care of loading contrastive model when
    # resuming training
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
    file_handler = logging.FileHandler(log_dir + '/' + name + ".log")
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_output_shape(model, image_dim):
    """
        Helper function to find the output shape of models
    """
    return model(torch.rand(*(image_dim))).data.shape


def get_checkpoints(path, match=None, white_list=[]):
    """
        Gets the checkpoints in the given directory

        Args:
            
            path (`string`):

                The dir to look for checkpoints in.

            match (`string`):
                
                The string to match checkpoints on. 
                Ex: given 'ckpt', it will return any file containing 'ckpt'
                If not given all files with be returned

            white_list: (`list`):

                If match is given, then whitelist will only return files
                that contain a string in this list. This is useful to only
                ask for certain checkpoints.

                Ex: you only want checkpoints 1 and 3, then checkpoint 2 will
                be skiped if given a list of [1, 3].
    """
    if match is None:
        checkpoints = [filename for filename in os.listdir(path)]
    else:
        checkpoints = [filename for filename in os.listdir(path) if match in filename]
        if white_list:
            checkpoints = [filename for filename in checkpoints
                           if any(checkpoint in filename for checkpoint in white_list)]
    checkpoints.sort()

    return checkpoints


def get_history(path):
    """
        Loads the loss history of the given dir.
    """
    if os.path.isfile(path):
        history = np.load(path)
        start_idx = history.shape[0]
        return history.tolist(), start_idx
    else:
        return [], 0


def save_zarr(data, zarr_file, ds, roi, voxel_size=(1, 1, 1), 
              num_channels=1, dtype=None, fit_voxel=False):
    """
        Helper function to save_zarr files using daisy.

        Args:
            
            data (`numpy array`):

                The data that you want to save.

            zarr_file (`string`):

                The zarr file you want to save to.

            ds (`string`):

                The dataset that the data should be saved as.

            roi (`daisy.Roi` or `list-like`):

                The roi to save the datset as.

            voxel_size (`tuple`, default=(1, 1, 1)):

                The voxel size to save the dataset as.

            num_channels (`int`, default=1):

                How many channels the data has. 
                Note: Daisy only supports saving zarrs with a single
                channel dim, so (num_channels, roi) is the only possible
                shape of the data.

            dtype (`numpy dtype`, optional):
                
                The datatype to save the data as

            fit_voxel (`bool`):
                
                If true then the roi will be multiplied by the voxel_size.
                This is useful if the ROI is in unit voxels and you want it
                to be in world units.


    """
    if not isinstance(roi, daisy.Roi):   
        roi = daisy.Roi([0 for d in range(len(roi))],
                        roi)

    if fit_voxel:
        roi = roi * voxel_size

    if dtype is None:
        dtype = data.dtype

    dataset = daisy.prepare_ds(zarr_file,
                               ds_name=ds,
                               total_roi=roi,
                               voxel_size=voxel_size,
                               dtype=data.dtype,
                               num_channels=num_channels)
    
    if roi.dims() > len(data.shape) and num_channels == 1:
        data = np.squeeze(data, 0)
    dataset.data[:] = data




