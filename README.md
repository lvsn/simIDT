## Getting Started (SAM-6D)

This is a fork of http://github.com/JiehongLin/SAM-6D/tree/main. Here is their getting started section. This section explains how to perform pose estimation with SAM-6D

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/JiehongLin/SAM-6D.git
```
Install the environment and download the model checkpoints:
```
cd SAM-6D
sh prepare.sh
```
We also provide a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for convenience.

### 2. Evaluation on the custom data
```
# set the paths
export CAD_PATH=Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=Data/Example/outputs         # path to a pre-defined file for saving results

# run inference
cd SAM-6D
sh demo.sh
```

### Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Getting started (Ours)

This section will explain how to do the simultaneous identification and tracking process.

### Preparation

Install the requirements in `requirements.txt`. You will also have to install the pose estimation module and the identification and segmentation module with :
```
cd SAM-6D/Instance_Segmentation_Model
pip install -e .
cd SAM-6D/Pose_Estimation_Model
pip install -e .
```
You might also need :
- [DeepTrack](https://github.com/Charles23R/deep_6dof_tracking)
- [FoundationPose](https://github.com/NVlabs/FoundationPose)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)

### Dataset structure

Here is an example of file structure for the objects database from which to identify/track.

```
database/
├── objects
│   ├── object0001/
│   │    ├── texture.jpg (or .png or .bmp)
│   │    ├── model.obj
│   │    └── model.mtl
│   │
│   ├── object0002/
│   │    ├── texture.jpg (or .png or .bmp)
│   │    ├── model.obj
│   │    └── model.mtl
│   ├── ...
└── camera.json
```

`camera.json` should look like this :
```
{"cam_K": [400.507, 0.0, 355.37, 0.0, 400.507, 199.815, 0.0, 0.0, 1.0], "depth_scale": 1.0}
```

The 3D models should be centered and the scale should be in mm (although the code should be able to deal with the latter).

### Data Pre-processing

First of all, generate the renders for the objects by running `render_multi.sh` and changing the `objects_path` variable to the path of your objects database.

Then, run `sample_cads.py` after changing the `objects_dir` variable to sample the CADs and extract features for the pose estimation model.

Then, run `compute_features.sh` with the right `DATA_PATH` to extract features necessary for the identification module.

Finally, run `Render/compute_scale.py` with the right `data_dir` argument to compute and save the scales of the different objects in the database.

### Running

To run the simultaneous identification and tracking system, simply run ``run_sequence_tracker_behave.sh`` with a path to a sequence organized like the [BEHAVE dataset](https://virtualhumans.mpi-inf.mpg.de/behave/) (single camera).