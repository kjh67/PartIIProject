import torch
import argparse
import os
import cv2
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from gaussian_splatting.scene import Scene
from gaussian_splatting.gaussian_renderer import GaussianModel
from gaussian_splatting.gaussian_renderer import render as render_gaussians
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args


def render_colmap_dir_splat(modelparams, iteration, pipelineparams):
    # SOURCE PATH is for colmap data,
    # MODEL PATH is for trained gaussian model location
    # IMAGE PATH should be for images associated with the colmap data

    print("Rendering test images")

    with torch.no_grad():
        gaussians = GaussianModel(modelparams.sh_degree)
        test_scene = Scene(modelparams, gaussians, load_iteration=iteration, shuffle=False)

        background=torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

        output_path = os.path.join(modelparams.model_path, "test", f"splat_iter{iteration}_renders")
        os.makedirs(output_path, exist_ok=True)
        assert(os.path.exists(output_path))

        for camera in test_scene.getTrainCameras():
            rendering = np.array(render_gaussians(camera, gaussians, pipelineparams, background)["render"].cpu().detach())
            print(rendering)
            print(rendering.shape)
            filename = camera.image_name + ".jpg"
            image_path = os.path.join(output_path, filename)
            print(image_path)
            written = cv2.imwrite(image_path, rendering)
            print(f"Rendered {camera.image_name} {written}")

    return output_path


def render_colmap_dir_nerf():
    return -1


def calculate_metrics(gt_folder, render_folder, output_file):
    """Assumes that the ground truth and render images to be compared have the same names"""

    psnrs = []
    ssims = []
    mses = []

    with os.scandir(gt_folder) as images:
        for image in images:
            # Load ground truth image
            gt_image = cv2.imread(image.path)

            # Load render
            render_image = cv2.imread(os.path.join(render_folder, image.name))

            # Compare, and record results
            psnrs.append(peak_signal_noise_ratio(gt_image, render_image))
            ssims.append(structural_similarity(gt_image, render_image))
            mses.append(mean_squared_error(gt_image, render_image))

    # Write results to the output file
    with open(output_file, 'w') as f:
        f.write(psnrs)
        f.write(ssims)
        f.write(mses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Computes PSNR, SSIM, and MSE for trained NVS models. Model must either be a NeRF or Gaussian Splat"""
    )
    # Gives the parameters --model_path, --source_path, --images
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--nerf", action='store_true')
    parser.add_argument("--splat", action='store_true')
    parser.add_argument("--output_file", type=str, default=None, help="File where outputs should be saved")
    parser.add_argument("--iteration", type=int, help="Evaluate model saved after this many iterations")

    args = get_combined_args(parser)

    if args.splat:
        output_dir = render_colmap_dir_splat(model.extract(args), args.iteration, pipeline.extract(args))

    if args.nerf:
        output_dir = render_colmap_dir_nerf()

    calculate_metrics(args.images, output_dir, args.output_file)
