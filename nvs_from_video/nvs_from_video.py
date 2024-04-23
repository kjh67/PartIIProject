import argparse
import subprocess
import os

from nvs_from_video.video_processing_utils import run_preprocessing


def process_splat(tgt, eval):
    # run gaussian optimisation, outputting into new folder
    if eval:
        colmap_path = os.path.join(tgt, "train", "colmap_path")
        image_path = os.path.join(tgt, "train", "images")
    else:
        colmap_path = os.path.join(tgt, "colmap_path")
        image_path = os.path.join(tgt, "frames")

    splat_processing = subprocess.run(["python", "gaussian-splatting/train.py",
                                       "-s", colmap_path, 
                                       "'m", tgt,
                                       "--images", image_path])
    if splat_processing.returncode != 0:
        print("Error processing gaussian splat")
        quit(code=1)

    if eval:
        pass
        # INSERT EVALUATION ACTIVITY HERE: render all frames, check PSNR for testing and training


def process_nerf(tgt, eval):
    if eval:
        colmap_path = os.path.join(tgt, "train", "colmap_path")
        image_path = os.path.join(tgt, "train", "images")
    else:
        colmap_path = os.path.join(tgt, "colmap_path")

    nerf_processing = subprocess.run(["ns-train", "nerfacto", "colmap", 
                                      "--data", tgt,
                                      "--images-path", image_path,
                                      "--colmap_path", colmap_path])
    if nerf_processing.returncode != 0:
        print("Error processing NeRF")
        quit(code=1)

    if eval:
        pass


if __name__ == "__main__":
    # check args

    parser = argparse.ArgumentParser(
        description="""
            Processes 360 video, stored in an equirectangular projection, to produce 
            either a NeRF or Gaussian Splat representation. COLMAP is used internally 
            for camera position estimation on each frame of the video used for training 
            and evaluation.""")
    parser.add_argument(
        '--reconstruction_type', type=str, required=True, choices=['nerf', 'splat'],
        help="Type of NVS representation to generate: either 'nerf' or 'splat")
    parser.add_argument(
        '--source_path', type=str, required=True,
        help="Path to raw 360 video")
    parser.add_argument(
        '--target_path', type=str, 
        help="Path to location where outputs should be saved")
    parser.add_argument(
        '--skip_framegen', '-sf', action='store_true',
        help="If active, the preprocessing stage will assume that frames have already \
            been extracted from the videos in the source directory into a 'frames' \
            folder in the target directory.")
    parser.add_argument(
        '--skip_colmap', '-sc', action='store_true',
        help="If active, preprocessing using COLMAP will be skipped. Only turn this on \
            when generating Gaussian Splats if COLMAP output has previously been \
            generated on the source data")
    parser.add_argument(
        '--colmap_map_only', '-cmo', action='store_true'
    )
    parser.add_argument(
        '--frame_sample_period', '-fp', type=int,
        help="Frequency at which frames are sampled from the source videos at the frame \
            extraction state. Default value 30 (corresponding to 1 sample per second \
            for 30fps video)."
    )
    parser.add_argument(
        '--eval', action='store_true'
    )

    args = parser.parse_args()
    tgt = run_preprocessing(args.source_path, args.target_path, args.skip_framegen, args.skip_colmap, args.frame_sample_period, args.colmap_map_only, args.eval)

    if args.reconstruction_type == "nerf":
        process_nerf(tgt, args.eval)
    elif args.reconstruction_type == "splat":
        process_splat(tgt, args.eval)
