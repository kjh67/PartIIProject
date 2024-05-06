import torch
import argparse
import os
import cv2
import numpy as np
import subprocess

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from gaussian_splatting.scene import Scene
from gaussian_splatting.gaussian_renderer import GaussianModel
from gaussian_splatting.gaussian_renderer import render as render_gaussians

from renderer.gauss_renderer import GaussianRenderer

from evaluation.eval_utils import ModelMetrics


class SplatModelParameters():
    def __init__(self, colmap_source_path, model_path, image_path):
        self.sh_degree = 3
        self.source_path = colmap_source_path
        self.model_path = model_path
        self.images = image_path
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False

class SplatPipelineParameters():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def eval_splats_cuda(project_dir, iterations):
    """Renders evaluation image, and then calls the metric calculator"""

    modelparams = SplatModelParameters(os.path.join(project_dir, "splat", "eval", "colmap"), 
                                       os.path.join(project_dir, "splat"),
                                       os.path.join(project_dir, "frames"))

    for iteration in iterations:
        print(f"Rendering splat iteration {iteration}")
        with torch.no_grad():
            gaussians = GaussianModel(modelparams.sh_degree)
            test_scene = Scene(modelparams, gaussians, load_iteration=iteration, shuffle=False)

            background=torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

            output_path = os.path.join(project_dir, "results", "renders", f"splat{iteration}")
            os.makedirs(output_path, exist_ok=True)

            for camera in test_scene.getTrainCameras():
                rendering = np.moveaxis(np.array(render_gaussians(camera, gaussians, SplatPipelineParameters(), background)["render"].cpu().detach()), 0, -1) * 255
                filename = camera.image_name + ".jpg"
                image_path = os.path.join(output_path, filename)
                written = cv2.imwrite(image_path, cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR))
        print("Rendering complete")
        calculate_metrics("splat", iteration, os.path.join(project_dir, "frames"), output_path, os.path.join(project_dir, "results", "render_metrics"))


def render_colmap_dir_splat_openGL(ply_path, colmap_source_path, output_path):

    print("Generating camera poses")

    # First; get camera positions as 4x4 matrices
    cameras = np.array()

    # Set up renderer using the correct fov, image size, etc
    renderer = GaussianRenderer()

    # for each model matrix (corresponding to camera); render and save image
    for camera in cameras:
        renderer.update_modelview(camera)
        renderer.render()
        image_name = '' + '.jpg'
        renderer.save_frame()


def eval_nerfs(project_dir, iterations):
    for iteration in iterations:
        # locate folder where the config.yml file will be located
        path_top = os.path.join(project_dir, "nerf", "unnamed", "nerfacto")
        timestamp_folder = os.scandir(path_top)[0]
        config_path = os.path.join(path_top, timestamp_folder)
    
        print(f"Rendering nerf iteration {iteration}")
        output_path = os.path.join(project_dir, "results", "renders", f"nerf{iteration}")
        render_command = ["ns-render", "dataset", "--load_config", os.path.join(config_path, "config.yml"), "--output_path", output_path]
        subprocess.run(render_command)
        print("Rendering complete")
        calculate_metrics("nerf", iteration, os.path.join(project_dir, "nerf", "images"), os.path.join(output_path, "rgb"), os.path.join(project_dir, "results", "render_metrics"))


def calculate_metrics(mode, iteration, gt_folder, render_folder, output_folder):
    """Assumes that the ground truth and render images to be compared have the same names"""

    psnrs = []
    ssims = []
    mses = []

    with os.scandir(render_folder) as images:
        for image in images:
            # Load rendered image
            render_image = cv2.imread(image.path)

            # Load render; (w,h,c) format
            gt_image = cv2.imread(os.path.join(gt_folder, image.name))

            # Calculate metrics
            psnrs.append(peak_signal_noise_ratio(gt_image, render_image))
            ssims.append(structural_similarity(gt_image, render_image, channel_axis=2))
            mses.append(mean_squared_error(gt_image, render_image))

    metrics = ModelMetrics(mode, iteration, psnrs, ssims, mses)
    metrics.save_metrics(output_folder)
    print("Metrics calculated and saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Computes PSNR, SSIM, and MSE for trained NVS models found in the provided folders."""
    )

    parser.add_argument("target_folders", nargs="+", help="Project folders to search for trained models")
    parser.add_argument("--iterations_to_evaluate", nargs="+", default=[30000],
                        help="Specify which iteration to generate renders (and metrics) for")
    parser.add_argument("--skip_nerfs", action='store_true')
    parser.add_argument("--skip_splats", action='store_true')

    args = parser.parse_args()

    for target in args.target_folders:
        if not args.skip_nerfs:
            if os.path.exists(os.path.join(target, "nerf")):
                eval_nerfs(target, args.iterations_to_evaluate)
        if not args.skip_splats:
            if os.path.exists(os.path.join(target, "splat")):
                eval_splats_cuda(target, args.iterations_to_evaluate)
