import os
import cv2
import numpy as np
from os.path import isfile, join

from tqdm import tqdm

def main(args):
    if args.image_dir is None:
        raise ValueError('input image folder is not given')

    files = [f for f in os.listdir(args.image_dir) if isfile(join(args.image_dir, f))]
    files.sort()
    frame_array = []

    fps = 30

    for i in tqdm(range(len(files))):
        filename= args.image_dir + '/' + files[i]
        try:
            img = cv2.imread(filename)
            height, width, layers = img.shape
        except:
        #reading each files
            continue
        size = (width, height)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(args.output_dir, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

    print('Done')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-id', '--image_dir', type=str, default=None, help='Path to the video file')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='Path to the video file')

    args = parser.parse_args()
    main(args)
