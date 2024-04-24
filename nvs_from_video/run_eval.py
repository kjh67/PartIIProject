import torch
import importlib
import argparse
import os
import cv2

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

gaussian_scene = importlib.import_module("gaussian-splatting/scene")
gaussian_renderer = importlib.import_module("gaussian-splatting/gaussian_renderer")
gaussian_arguments = importlib.import_module("gaussian-splatting/arguments")

# need to pass args as a ModelParas object to the scene contructor
# need TWO SETS of model params; one for the trained scene, and one for the eval scene


def render_colmap_dir_splat(modelparams, iteration, pipelineparams):
    # SOURCE PATH is for colmap data,
    # MODEL PATH is for trained gaussian model location
    # IMAGE PATH should be for images associated with the colmap data

    print("Rendering test images")

    with torch.no_grad():
        gaussians = gaussian_renderer.GaussianModel(modelparams.sh_degree)
        test_scene = gaussian_scene.Scene(modelparams, gaussians, load_iteration=iteration, shuffle=False)

        background=torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

        output_path = os.path.join(modelparams.model_path, "test", f"splat_iter{iteration}_renders")
        os.makedirs(output_path, exist_ok=True)

        for camera in test_scene.getTrainCameras():
            rendering = gaussian_renderer.render(camera, gaussians, pipelineparams, background)["render"]

            cv2.imwrite(os.path.join(output_path, camera.image_name), rendering)
            print(f"Rendered {camera.image_name}")

    return output_path


def render_colmap_dir_nerf():
    return -1


def save_data():
    pass


def calculate_metrics(gt_folder, render_folder):
    """Assumes that the ground truth and render images to be compared have the same names"""

    psnrs = []
    ssims = []
    mses = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Computes PSNR, SSIM, and MSE for trained NVS models. Model must either be a NeRF or Gaussian Splat"""
    )
    # Gives the parameters --model_path, --source_path, --images
    model = gaussian_arguments.ModelParams(parser, sentinel=True)
    pipeline = gaussian_arguments.PipelineParams(parser)

    parser.add_argument("--nerf", action='store_true')
    parser.add_argument("--splat", action='store_true')
    parser.add_argument("--output_file", type=str, default=None, help="File where outputs should be saved")
    parser.add_argument("--iteration", type=int, help="Evaluate model saved after this many iterations")

    args = gaussian_arguments.get_combined_args(parser)

    if args.splat:
        output_dir = render_colmap_dir_splat(model.extract(args), args.iteration, pipeline.extract(args))

    if args.nerf:
        output_dir = render_colmap_dir_nerf()
