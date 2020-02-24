"""Object detector inference
@kv
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from os.path import isfile, join
from tqdm import tqdm

import time

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result
import mmcv


def plot_and_save_bboxes(img_arr, bboxes, out_path):
    img_h, img_w, _ = img_arr.shape
    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(img_arr.astype(np.uint8))
    for bbox in bboxes:
        bbox[0] = bbox[0] * img_h
        bbox[1] = bbox[1] * img_w
        bbox[2] = bbox[2] * img_h
        bbox[3] = bbox[3] * img_w
        ymin, xmin, ymax, xmax = bbox
        x = int(xmin)
        y = int(ymin)
        w = int(xmax - xmin)
        h = int(ymax - ymin)
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1,
            edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
    plt.savefig(out_path)
    plt.clf()

def main(args):

    if args.image_dir is None:
        raise ValueError('No input!')
    image_filenames = [f for f in os.listdir(args.image_dir) if isfile(join(args.image_dir, f))]
    image_paths = [join(args.image_dir, f) for f in image_filenames]

    if not os.path.exists(args.output_dir) and not args.video_output:
        os.makedirs(args.output_dir)

    config_file = args.config_path
    checkpoint_file = args.model_dir

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    inference_times = []
    """TODO:
        - Do not save images anymore. (need cooperate with viewer)
        - Save inference results as list of dict to numpy 
            a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        - Support video in, video out.
    """
    if args.video_format:
        video_out = cv2.VideoWriter(args.output_dir, cv2.VideoWriter_fourcc(*'DIVX'), args.fps, (640, 480))

    for img_path in tqdm(image_paths):
        start_time = time.time()
        result = inference_detector(model, img_path)
        end_time = time.time()
        inference_times.append(end_time - start_time)
        output_path = join(args.output_dir, os.path.basename(img_path))
        #plot_and_save_bboxes(raw_img, ret['detection_boxes'], output_path)
        #print(model.CLASSES)
        if args.video_output:
            img = show_result(img_path, result, [model.CLASSES], out_file=None, show=False)
            video_out.write(img)
        else:
            show_result(img_path, result, [model.CLASSES], out_file=output_path, show=False)

    print(np.mean(inference_times))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Path to the model folder')
    parser.add_argument('-md', '--model_dir', type=str, default=None, help='Path to the model folder')
    parser.add_argument('-id', '--image_dir', type=str, default=None, help='Input image directory')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('-vout', '--video_output', action='store_true', help='Use video as output format')
    parser.add_argument('--fps', type=int, default=6, help='FPS of output video')
    args = parser.parse_args()
    main(args)
