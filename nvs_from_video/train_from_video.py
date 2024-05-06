import argparse
import subprocess
import os

from nvs_from_video.video_processing_utils import run_preprocessing


def process_splat(tgt):
    # run gaussian optimisation, outputting into new folder
    colmap_path = os.path.join(tgt, "splat", "train", "colmap")
    image_path = os.path.join(tgt, "frames")
    model_path = os.path.join(tgt, "splat")

    splat_processing = subprocess.run(["python", "-m", "gaussian_splatting.train",
                                       "-s", colmap_path, 
                                       "-m", model_path,
                                       "--images", image_path,
                                       "--test_iterations", "-1",
                                       "--save_iterations", "5000", "7000", "10000", "15000", "20000", "25000", "30000"])
    if splat_processing.returncode != 0:
        print("Error processing gaussian splat")
        quit(code=1)


def process_nerf(tgt):
    model_path = os.path.join(tgt, "nerf")

    nerf_processing = subprocess.run(["ns-train", "nerfacto",
                                      "--output-dir", model_path,
                                      "--experiment-name", "",
                                      "--timestamp", "",
                                      "--steps-per-save", "5000",
                                      "--max-num-iterations", "30001",
                                      "--save-only-latest-checkpoint", "False",
                                      "--logging.steps-per-log", "100",
                                      "--viewer.websocket-port", "7007",
                                      "--viewer.make-share-url", "True",
                                      "colmap", 
                                      "--data", model_path,
                                      "--eval-mode", "filename",
                                      "--downscale-factor", "1"])
    if nerf_processing.returncode != 0:
        print("Error processing NeRF")
        quit(code=1)


if __name__ == "__main__":

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
        '--skip_colmap_featureextract', action='store_true'
    )
    parser.add_argument(
        '--skip_colmap_featurematch', action='store_true'
    )
    parser.add_argument(
        '--frame_sample_period', '-fp', type=int, default=30,
        help="Frequency at which frames are sampled from the source videos at the frame \
            extraction state. Default value 30 (corresponding to 1 sample per second \
            for 30fps video)."
    )
    parser.add_argument(
        '--train_proportion', type=float, default=0.8,
        help="Proportion of extracted frames to be used for training (remainder reserved for evaluation)"
    )
    parser.add_argument(
        "--colmap_exhaustive_match", action='store_true',
        help="Will use exhaustive matching during COLMAP processing. Not recommended for more than a few hundred images"
    )
    parser.add_argument(
        "--colmap_vocabtree_match", action='store_true'
    )
    parser.add_argument(
        "--colmap_vocabtree_location", type=str, default="./nvs_from_video/vocab_tree.bin"
    )
    parser.add_argument(
        "--colmap_use_gpu", action='store_const', const="1", default="0"
    )

    args = parser.parse_args()
    tgt = run_preprocessing(args)

    if args.reconstruction_type == "nerf":
        process_nerf(tgt)
    elif args.reconstruction_type == "splat":
        process_splat(tgt)
