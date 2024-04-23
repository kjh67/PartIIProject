import os
import cv2
import subprocess
import numpy as np
import shutil

from nvs_from_video.map_equi_pinhole import map_equi_pinhole
from colmap.scripts.python.read_write_model \
    import read_cameras_binary, write_cameras_binary, \
        read_images_binary, write_images_binary, \
        read_points3D_binary, write_points3D_binary


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
    with open(os.path.join(tgt, "colmap_output", "database.db"), "w") as _: pass

    # 3. set up splatting output folder
    os.mkdir(os.path.join(tgt, "model_output"))

    return tgt


def retrieve_colmap_information(image_ids, all_images, all_cameras, all_points3D):
    images = {}
    cameras = {}
    points3D = {}
    image_names = []
    for im_id in image_ids:
        images[im_id] = all_images[im_id]
        cam_id = images[im_id].camera_id
        cameras[cam_id] = all_cameras[cam_id]
        points3D_ids = images[im_id].point3D_ids
        for point_id in points3D_ids:
            # Many of the point_ids are -1, signifying no point
            if point_id != -1:
                points3D[point_id] = all_points3D[point_id]
        image_names.append(images[im_id].name)
    return images, cameras, points3D, image_names


def train_test_split(source_directory, train_proportion=0.8):
    """Takes an output directory, containing directories 'colmap_output' and 
    'frames', and creates separate 'train' and 'test' directories"""

    print("Generating training and testing datasets")

    # Make directories: colmap_output/test/sparse/0, and colmap_output/train/sparse/0
    source_colmap_directory = os.path.join(source_directory, "colmap_output", "sparse", "0")
    train_colmap_directory = os.path.join(source_directory, "train", "colmap_output", "sparse", "0")
    test_colmap_directory = os.path.join(source_directory, "test", "colmap_output", "sparse", "0")

    if os.path.exists(train_colmap_directory):
        print("Training and testing sets previously generated")
        return

    os.makedirs(os.path.join(train_colmap_directory))
    os.makedirs(os.path.join(test_colmap_directory))

    # Get list of all COLMAPed images; returned as dictionaries
    images = read_images_binary(os.path.join(source_colmap_directory, "images.bin"))
    cameras = read_cameras_binary(os.path.join(source_colmap_directory, "cameras.bin"))
    points3D = read_points3D_binary(os.path.join(source_colmap_directory, "points3D.bin"))

    print("COLMAP information read")

    # Take a subset of shuffled images for training, and the rest for testing
    image_ids_shuffled = list(images.keys())
    np.random.shuffle(image_ids_shuffled)
    train_image_ids = image_ids_shuffled[:int(len(image_ids_shuffled)*train_proportion)]
    test_image_ids = image_ids_shuffled[int(len(image_ids_shuffled)*train_proportion):]

    # Retrive the COLMAP information into dicts
    train_images, train_cameras, train_points3D, train_image_names = retrieve_colmap_information(
        train_image_ids, images, cameras, points3D
    )
    test_images, test_cameras, test_points3D, test_image_names = retrieve_colmap_information(
        test_image_ids, images, cameras, points3D
    )

    # Write back information to relevant directories
    write_images_binary(train_images, os.path.join(train_colmap_directory, "images.bin"))
    write_images_binary(test_images, os.path.join(test_colmap_directory, "images.bin"))
    write_cameras_binary(train_cameras, os.path.join(train_colmap_directory, "cameras.bin"))
    write_cameras_binary(test_cameras, os.path.join(test_colmap_directory, "cameras.bin"))
    write_points3D_binary(train_points3D, os.path.join(train_colmap_directory, "points3D.bin"))
    write_points3D_binary(test_points3D, os.path.join(test_colmap_directory, "points3D.bin"))

    print("COLMAP information written")

    # Copy images into test and train folders
    source_image_directory = os.path.join(source_directory, "frames")
    train_image_directory = os.path.join(source_directory, "train", "images")
    test_image_directory = os.path.join(source_directory, "test", "images")

    os.mkdir(train_image_directory)
    os.mkdir(test_image_directory)

    for image in train_image_names:
        shutil.copy(os.path.join(source_image_directory, image), os.path.join(train_image_directory, image))
    print("Finished writing images to training directory")
    for image in test_image_names:
        shutil.copy(os.path.join(source_image_directory, image), os.path.join(test_image_directory, image))
    print("Finished writing images to testing directory")

    print("Training and testing datasets generated.")


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
        print("No videos found")
        return

    cont, frame = video.read()
    frame_number = 0
    while cont:
        if frame_number%frequency== 0:
            for n, mapping in enumerate(mappings):
                fname = '/frame' + "{:04d}".format((frame_number // frequency)*9 + n) + '.jpg'
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


def process_colmap(image_dir, target_dir, map_only=False, vocabtree_location="./nvs_from_video/vocab_tree.bin"):
    """src is the directory containing extracted frames; tgt is 
    the directory in which to output the COLMAP database and 
    reconstruction files"""

    def stop_if_failed(subprocess_obj):
        if subprocess_obj.returncode != 0:
            print("COLMAP Processing failed")
            quit(code=1)

    # assume that colmap has already been installed; call the functions as a subprocess rather than using the pycolmap module
    db_path = os.path.join(target_dir,"database.db")

    if not map_only:
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
            feature_matching_command += ["--SequentialMatching.loop_detection", "1",
                                        "--SequentialMatching.vocab_tree_path", vocabtree_location]

        feature_matching = subprocess.run(feature_matching_command)
        stop_if_failed(feature_matching)

    # perform incremental mapping to register image locations
    incremental_mapping = subprocess.run(["colmap", "mapper",
                                          "--database_path", db_path,
                                          "--image_path", image_dir,
                                          "--output_path", os.path.join(target_dir,"sparse"),
                                          "--Mapper.max_num_models", "1"])
    stop_if_failed(incremental_mapping)

    print("COLMAP processing complete")


def run_preprocessing(src, tgt, frames_exist=False, skip_colmap=False, frame_sample_period=30, colmap_map_only=False, split_train_test=True, train_proportion=0.8):
    
    try:
        tgt = directory_setup(src, tgt)
    except DirectorySetupError as e:
        print(e.message)
        quit(code=1)
    except FileExistsError as e:
        print("Filespace previously set up")

    # extract frames into new folder
    if not frames_exist:
        try:
            extract_frames(src, os.path.join(tgt, "frames"), frame_sample_period)
        except FileExistsError as f:
            print("Failed to extract frames from video")
            quit(code=1)

    # run colmap, placing output in the folder set up earlier
    # TODO: check whether colmap already run for this dataset, and also skip automatically
    if not skip_colmap:
        process_colmap(os.path.join(tgt, "frames"), os.path.join(tgt, "colmap_output"), colmap_map_only)

    if split_train_test:
        train_test_split(tgt, train_proportion)

    return tgt
