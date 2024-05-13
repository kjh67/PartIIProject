# PartIIProject

## Project Dependencies

To ensure that all submodule repositories are present, this project should be cloned with the `--recurse_submodules` flag set

The NVS pipeline required the COLMAP library to be built and installed; this can be accomplished using the code below. For more details please see https://colmap.github.io/install.html.
```
cd ./colmap
!mkdir build
cd ./build
!cmake .. -GNinja
!ninja
!sudo ninja install
```

Additionally, dependencies from the Gaussian Splatting repository must be installed as follows:
```
cd ./gaussian_splatting
!pip install -q ./submodules/diff-gaussian-rasterization
!pip install -q ./submodules/simple-knn
```

The following other project dependencies should be installed using pip:
- OpenCV (package name cv2)
- NumPy
- SciPy
- Pygame
- Plyfile
- Matplotlib
- Nerfstudio


## Using the NVS Pipeline

The NVS pipeline can be run from the repository directory using the following command:

```
python -m nvs_from_video.train_from_video --reconstruction_type <reconstruction type> --source_path <path to raw data>
```

The full set of command line arguments for controlling the pipeline are as follows:
- `--reconstruction_type`: Type of NVS representation to generate: either 'nerf' or 'splat'.
- `--source_path`: Path to directory containing 360 video.
- `--target_path`: The directory where outputs should be saved. If none is provided, a new folder named '<source_path>_output' will be created.
- `--skip_framegen` (flag): If active, preprocessing using COLMAP will be skipped. Only turn this on if COLMAP output has previously been generated for the source data.
- `--frame_sample_period`: Frequency at which frames are sampled from the source videos at the frame extraction state. Default value 30 (corresponding to 1 sample per second for 30fps video).
- `--train_proportion`: Proportion of extracted frames to be used for training (remainder reserved for evaluation). Defaults to 0.9.
- `--skip_colmap` (flag): If active, preprocessing using COLMAP will be skipped. Only use if COLMAP output has previously been generated on the source data.
- `--skip_colmap_featureextract` (flag): If active, will cause the COLMAP pipeline to skip feature extraction.
- `--skip_colmap_featurematch` (flag): If active, will cause the COLMAP pipeline to skip feature matching.
- `--colmap_vocabtree_match` (flag): If set, vocabulary tree matching will be used during COLMAP processing. Sequential matching is used by default.
- `--colmap_vocabtree_location`: Path to the vocabulary tree to be used by COLMAP. Defaults to nvs_from_video/vocab_tree.bin.
- `--colmap_use_gpu`: Turns on GPU acceleration for applicable COLMAP routines.

## Evaluating NVS Pipeline Results

The script to evaluate all trained NVS models found within a set of target folders can be run using the command:
```
python -m evaluation.eval <target folders>
```
Further command line options can be viewed by calling `python -m evalution.eval --help`.

The plotting.py script is also provided, which provides utilities for plotting evaluation results. The command line syntax can be viewed by running `python -m evaluation.plotting --help`.



# Running the Biking Game

The biking game is run from the command line using the following command:
```
python -m game.main <path to PLY file containing Gaussians to render>
```

Further command line options, including how to configure the constrained camera movement by providing a set of COLMAP points, can be viewed by running `python -m game.main --help`.
