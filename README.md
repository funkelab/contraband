# Contraband

Contraband is a code base built using Gunpowder and PyTorch to use recent contrastive learning techniques to find hierarchical data representations. Combine with few-shot learning techniques to allow fast, interactive learning on supervised tasks.
### Goals

Compare different embedding networks against baseline models, in terms of:
* accuracy
* training speed
for different amounts of training data.

Baselines:
* U-Net with same parameters as embedding network, directly trained on supervised task
* Just the segmentation head on supervised task

### General Setup of Train and Validation

Steps:

For the embedding network:
1. Training of embedding network

For each embedding network + baselines:
3. Training of the segmentation head
4. Validtion of segmentation head
5. Testing of segmentation head

Datasets:
For each $k$ (amount of training data), each dataset should be split into:
1. Train
2. Validate
3. Test
### Structure of experiments:
	expirements
	|--dataset name/
	   |--EXPn-short_desc-month-day-year/
	      |--param_dict.json
	      |__combination-0/
	      |   ...
	      |--combination-n
	         |--contrastive
		        |--histpy.npy
		        |--checkpionts/
		        |--snapshots/
		 |--seg/
	            |--copntrastive_ckpt1/
		    |...
		    |--contrastive_ckptn/
		       |--histpy.npy
		       |--checkpoints/
		       |--snapshots/
		       |--metrics/
		       |  |--metrics_$checkpoint_num.csv
		       | --samples
		       |  |--sample_$checkpoint_num.zarr
		 |--emb (optional)
		    |--contrastive_ckptn
		       |--dataset type e.g. train,validate,test
		          |--raw_embs.zarr
#### What are combinations?
Combinations refer to the hyper parameters that each run uses. Each combination is a unique combination of the hyperparmetrs specified in the param_dict.json in each experiment.

#### param_dicts
Each experiment needs to have a param_dict.json to set up the model and pipelines with the correct settings. See bottom for example.

Most things are in lists, this is because a grid will be made using a combination of values within each dict. e.g. within `model`, `contrastive`, and `segmentation` all combinations of values will be made.

There are two different number of combinations, one with the contrastive and model parameters, and the other using the segmentation parameters. Looking back to the expirements directory structure, the first level of combinations represents these contrastive and model combinations. Here the combinations for the contrastive pipeline and models are combined. So given 3 contrastive combinations and 3 model combinations we will have 9 total combinations. This makes sense because the model parameters represent the base encoder which gets trained using the contrastive pipeline settings, so we would want to do a grid search over all of this.

The second level of combinations in the `set` dir, represent the segmentation combinations.

Generating these combinations is handled in `Trainer.py`

### Design

`ContrastiveVolumeNet` takes a base net, adds a projection head
* for training, passes two raw volumes through base net and projection head
* no inference needed, this is the base net

`SegmentationNet` takes a base net and a segmentation head
* Can take a raw volume, ebeddings, or embeddings and points.

Base nets:
* `UNet` from funlib, with respective arguments

Segmentation heads:
* `SimpleSegHead` this is just a simple MLP on top of the base encoder
* `SparseSegHead` This seg head is to train on sparse amounts of GT by using torch linear layers. It takes in embeddings and points and gives its predictions only on those points. Alternatively, you can give it a list of embeddings and it will give predictions directly on those. Can transform it's weights to convolutional layers during evaluation.

Pipelines:
All pipelines should work with 2D and 3D data.
* `Contrastive` This pipeline is to train using the contrastive structure and loss. It has two input branches for both views of the data. 
* `Segmentation` This pipeline is to train the segmentation head.
* `SparseSH` This pipeline trains the SparseSegHead on pre-computed embeddings
* `SparseBasline` This pipeline is to train an end to end baseline using the SparseSegHead. This essentially combines the functionally of the `Segmentation` and `SparseSH` pipelines.
### Scripts
Contraband is has 4 different scripts. 

#### `train_contrastive.py` script
This script is for contrastively training models.
This script uses the `contrastive` and `model` parameter dictionarys.
#### `train_seg.py` script
This script is for training the segmentation head (or baselines). This script uses the `seg` param dict to get number of combinations, but also
uses the `model` dict to make baseline model.
#### `validate.py` script
This script is used for make predictions on validation data and generation segmentations. Currently can only do watershed -> agglomeration to get results. Returns results as 
VOI Sum. Uses the segmentation param dict.
#### `make_embs.py` script
This script will generate the embeddings (h) for a dataset using a trained contrastive modelto be used layer in the sparse segmentation head.

### How to use
Run command 
```
python train.py 
			-dataset # what dataset dir in expirements to use
			-exp # The expirement number
			-mode # Which mode (contrastive, seg, val, emb)
			-index # What contrastive combination to run
			-checkpionts # What contrastive checkpoints to use while in seg, val, or emb mode.
```
Steps to train with SparseSH:

1. Run `generate_points.py`
2. Train contrastive model
3. Create embeddings (run in `emb` mode)
4. Train seg head (`seg` mode)

Example `param_dict.json`:
```
{
  "pipeline": "pipelines to use, check param_mapping.py to see what pipeline
			   names are avalible and what they map to",

  "save_embs":{ 
  	  # What dataset in given data file to use. Can specify multiple
	  "datasets": ["validate/raw"], 
	  # What data zarr to pull data from
      "data_file": ["data/ctc/Fluo-N2DH-SIM+.zarr"]
  },

  "model": [
     {
      "model": "unet", # Only supported model right now.
      "in_shape": [260, 260],

      "in_channels": 1,
      "num_fmaps": 12,
      "fmap_inc_factors": 6,
      "downsample_factors": [[2, 2], [2, 2], [2, 2]],
      "kernel_size_down": [[[3, 3], [3, 3]]],
      "kernel_size_down_repeated": 4,
      "kernel_size_up": [[[3, 3], [3, 3]]],
      "kernel_size_up_repeated": 3,
      "constant_upsample": true,

      "h_channels": 12, # size of intermediant embedding or `h`
      "contrastive_out_channels": 12 # The size of projection head embedding, should be the same as h_channels.
  },
  ...,
  {
	  # Can give multiple model specs. Grid of parameters will only be made over
	    the the number of model specs given. 
		Ex: given 3 model specs, there will be 3 combinations created.
  }
  ],

  "contrastive": {
      "data_file": ["data/ctc/Fluo-N2DH-SIM+.zarr"],
      "dataset": [["train/raw"]],
      "optimizer": ["adam"],
      "lr": [1e-5],
      "num_iterations": [10000],
      # What augmentations should be used
      "elastic": [true],
      "blur": [true],
      "simple": [true],
      "noise": [true],
      # The parameters for each augmentation
      "elastic_params": [{
		  "control_point_spacing": [1, 10, 10],
		  "jitter_sigma": [0.0, 0.1, 0.1],
	   	  "rotation_interval": [0, "pi / 2"]
      }],
      "blur_params": [{"sigma": [0, 1, 1]}],
      "simple_params": [{
                "mirror_only": [1, 2],
                "transpose_only":[1, 2]
      }],
      "noise_params": [{
	    "var": 0.000001
      }],
      
      # How much to scale the raw data by to make all values <=1
      "norm_factor": [0.25],
	  # How often to save the model to a checkpoint file
      "save_every": [5000],

	  # How many points you want per voxel
      "point_density": [0.01],
      # Temp parameter for SimCLR loss.
      "temperature": [0.05]
  },

  "seg": {
	  # Optional parameters for baseline
	  "basline": [false],
	  # Should include freeze_base if training baseline.
	  # If true this will freeze the base encoder.
	  # If false or left out the base_encoder will not be frozen
	  "freeze_base": [true], 
	  
      "data_file": ["data/ctc/Fluo-N2DH-SIM+.zarr"],
      "dataset": [{
	      "train": {"raw": "train/raw", "gt": "train/gt"},
	      "validate": {"raw": "embs", "gt": "validate/gt"}
      	}
      ],
      "batch_size": [4],
      "optimizer": ["adam"],
      "lr": [1e-5],
      "num_iterations": [10000],
      "blur": [true],
      "simple": [true],
      "noise": [true],
      "elastic": [true],
      "elastic_params": [{
		  "control_point_spacing": [1, 10, 10],
		  "jitter_sigma": [0.0, 0.1, 0.1],
		  "rotation_interval": [0, "pi / 2"]
      }],
      "blur_params": [{"sigma": [0, 1, 1]}],
      "simple_params": [{
          "mirror_only": [1, 2],
          "transpose_only":[1, 2]
      }],
      "noise_params": [{
	      "var": 0.000001
      }],
      # Which segmentation head to use, see param_mapping.py for specific names.
      "seg_head": ["Sparse"],
	  
	  # What thresholds to use for agglomeration
      "thresholds": [[0.5, 0.75]],
      "norm_factor": [0.25],

      "save_every": [5000],

	  # How many output channels the segmentation head should have
      "seg_out_channels": [2],

	  # If data gt has a background that needs to be predicted, then setting
	  # this to true will add extra seeds during watershed to help predict
	  # background better.
      "has_background": [true],

     # When using the sparse segmentation head, you can specify how many points
     # to use when training. 
      "num_points": [16, 32]
  }
}
```a
