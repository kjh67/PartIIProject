import cv2
import os
import sys
import numpy as np

from map_equi_pinhole import map_equi_pinhole


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


def extract_frames(src, tgt, frequency=200):
    """"""

    sources = os.scandir(src) # get list of video files in the source folder

    # generate mappings from the equirectangular image to pinhole images
    # plus and minus 25deg from level, and plus minus 45 each side
    mappings = []
    for yaw in range(-45,46,45):
        for pitch in range(-25, 26, 25):
            x_map, y_map = map_equi_pinhole(100,5760,2880,1600,1200,yaw, pitch)
            mappings.append([x_map.astype(np.float32), y_map.astype(np.float32)])

    print(f"No of pinholes extracted per equirec: {len(mappings)}")

    video, sources = get_next_video(sources)
    if not video:
        return

    cont, frame = video.read()
    frame_number = 0
    while cont:
        if frame_number%frequency== 0:
            # HERE insert extracting however many 'normal' images from the equirectangular projections
            for n, mapping in enumerate(mappings):
                fname = '/frame' + str(frame_number // frequency) + '_' + str(n) + '.jpg'
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


if __name__=="__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        AttributeError("Two args must be supplied: source video and target folder for frames")
        print('Error!')
    else:
        src = os.curdir +'/' + args[0]
        tgt = os.curdir +'/' + args[1]
        extract_frames(src,tgt)