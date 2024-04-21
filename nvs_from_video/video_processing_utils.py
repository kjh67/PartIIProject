import os
import cv2
import subprocess
import numpy as np

from nvs_from_video.map_equi_pinhole import map_equi_pinhole
from colmap.scripts.python.read_write_model import read_model, write_model


class DirectorySetupError(Exception):
    def __init__(self, msg):
        super().__init__()
        self.message = msg


def directory_setup(src, tgt):
    if not os.path.exists(src):
        raise DirectorySetupError("Error: Source directory does not exist")

    # ensure that a target folder exists
    if tgt is None:
        # get the source directory name, create a target called NAME_output
        tgt = os.path.join(os.path.dirname(src), os.path.basename(src)+'_output')
    if not os.path.exists(tgt):
        os.makedirs(tgt)

    # 1. set up frame extraction folder
    os.mkdir(os.path.join(tgt, "frames"))
    
    # 2. set up colmap output folder, including empty database and sparse folder
    os.makedirs(os.path.join(tgt, "colmap_output/sparse"))
    with open(os.path.join(tgt, "colmap_output", "database.db")) as _: pass

    # 3. set up splatting output folder
    os.mkdir(os.path.join(tgt, "model_output"))

    return tgt


def get_next_video(src_list):
    try:
        file = next(src_list)
        print(file.name)
        if file.name[-4:] == '.mp4':
            print('making video')
            video = cv2.VideoCapture(file.path)
            return video, src_list
        print('Failed to find mp4')
    except:
        print('source list empty')
        return None, None


def extract_frames(src, tgt, frequency=30):
    """Assumes input video at 30fps, downsampling to one frame per second by default"""

    sources = os.scandir(src) # get list of video files in the source folder

    # generate mappings from the equirectangular image to pinhole images
    # plus and minus 25deg from level, and plus minus 45 each side
    mappings = []
    for yaw in range(-45,46,45):
        for pitch in range(-25, 26, 25):
            x_map, y_map = map_equi_pinhole(100,5760,2880,1600,1200,yaw, pitch)
            mappings.append([x_map.astype(np.float32), y_map.astype(np.float32)])

    video, sources = get_next_video(sources)
    if not video:
        return

    cont, frame = video.read()
    frame_number = 0
    while cont:
        if frame_number%frequency== 0:
            for n, mapping in enumerate(mappings):
                fname = '/frame' + str((frame_number // frequency)*9 + n) + '.jpg'
                to_write = cv2.remap(frame, mapping[0], mapping[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
                cv2.imwrite((tgt + fname), to_write)

        if frame_number%(frequency*10) == 0:
            print(str(frame_number//frequency)+' frames extracted')
        frame_number += 1

        cont, frame = video.read()
        if not cont:
            # get the next video
            video.release()
            video, sources = get_next_video(sources)
            if video:
                cont = True
                print('Progressed to next video')

    cv2.destroyAllWindows()
    print('Extraction complete')


def process_colmap(image_dir, target_dir, vocabtree_location="./nvs_from_video/vocab_tree.bin"):
    """src is the directory containing extracted frames; tgt is 
    the directory in which to output the COLMAP database and 
    reconstruction files"""

    def stop_if_failed(subprocess_obj):
        if subprocess_obj.returncode != 0:
            print("COLMAP Processing failed")
            quit(code=1)


    db_path = os.path.join(target_dir,"database.db")

    # assume that colmap has already been installed; call the functions as a subprocess rather than using the pycolmap module
    feature_extraction = subprocess.run(["colmap", "feature_extractor",
                                         "--database_path", db_path,
                                         "--image_path", image_dir,
                                         "--ImageReader.camera_model", "PINHOLE"])
    stop_if_failed(feature_extraction)

    # since frames are ordered with overlap between consecutive frames, use sequential feature matching
    feature_matching_command = ["colmap", "sequential_matcher",
                                       "--database_path", db_path]

    # check to see whether a vocab tree is in the expected place; if so, add loop detection to the feature matching command
    if os.path.isfile(vocabtree_location):
        feature_matching_command += ["--SequentialMatching.loop_detection", 1,
                                       "--SequentialMatching.vocab_tree_path", vocabtree_location]

    feature_matching = subprocess.run(feature_matching_command)
    stop_if_failed(feature_matching)

    # perform incremental mapping to register image locations
    incremental_mapping = subprocess.run(["colmap", "mapper",
                                          "--database_path", db_path,
                                          "--image_path", image_dir,
                                          "--output_path", os.path.join(target_dir,"sparse"),
                                          "--Mapper.max_num_models", 1])
    stop_if_failed(incremental_mapping)

    print("COLMAP processing complete")


def run_preprocessing(src, tgt, frame_sample_period=30):
    # check whether a colmap dataset already exists for this set of parameters
    
    try:
        tgt = directory_setup(src, tgt)
    except DirectorySetupError as e:
        print(e.message)
        quit(code=1)

    # extract frames into new folder
    try:
        extract_frames(src, os.path.join(tgt, "frames"), frame_sample_period)
    except FileExistsError as f:
        print("Failed to extract frames from video")
        quit(code=1)

    # run colmap, placing output in the folder set up earlier
    # check whether colmap already run for this dataset
    process_colmap(src, os.path.join(tgt, "colmap_output"))

    return tgt
