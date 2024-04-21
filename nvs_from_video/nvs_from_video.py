import argparse
import subprocess
import os

from nvs_from_video.video_processing_utils import run_preprocessing


def splat_from_video(src, tgt, fe):

    tgt = run_preprocessing(src, tgt, fe)

    # run gaussian optimisation, outputting into new folder
    splat_processing = subprocess.run(["python", "gaussian-splatting/train.py",
                                       "-s", os.path.join(tgt, "colmap_output"), 
                                       "'m", tgt,
                                       "--images", os.path.join(tgt, "frames")])
    if splat_processing.returncode != 0:
        print("Error processing gaussian splat")
        quit(code=1)


def nerf_from_video(src, tgt, fe):

    tgt = run_preprocessing(src, tgt, fe)

    nerf_processing = subprocess.run(["ns-train", "nerfacto", "colmap", 
                                      "--data", tgt,
                                      "--images-path", os.path.join(tgt,"frames"),
                                      "--colmap_path", os.path.join(tgt, "sparse","0")])
    if nerf_processing.returncode != 0:
        print("Error processing NeRF")
        quit(code=1)


if __name__ == "__main__":
    # check args

    parser = argparse.ArgumentParser()
    parser.add_argument('reconstruction_type', type=str, help="Either 'nerf' or 'splat'")
    parser.add_argument('source_path', type=str, help="Path to raw 360 video")
    parser.add_argument('target_path', type=str, help="Path to location where outputs should be saved", nargs="?")
    parser.add_argument('--frames_exist', action='store_true')

    # add further options for frame sampling frequency etc

    args = parser.parse_args()

    if args.reconstruction_type.lower() == "nerf":
        nerf_from_video(args.source_path, args.target_path, args.frames_exist)
    elif args.reconstruction_type.lower() == "splat":
        splat_from_video(args.source_path, args.target_path, args.frames_exist)
    else:
        print("Reconstruction type must be either 'nerf' or 'splat'")
