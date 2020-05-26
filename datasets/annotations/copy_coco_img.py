
import os
import sys
from os.path import join
import json

from tqdm import tqdm
from pycocotools.coco import COCO

from shutil import copyfile


def main(args):
    if args.output_dir is None:
        raise ValueError('Output dir is not assigned')

    os.makedirs(args.output_dir, exist_ok=True)

    src_dir = args.image_dir
    dst_dir = args.output_dir

    coco = COCO(args.groundtruth_path)
    imgInfo = coco.loadImgs(coco.getImgIds())

    for img in tqdm(imgInfo):
        file_name = img['file_name']
        copyfile(
            join(src_dir, file_name),
            join(dst_dir, file_name)
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-gt', '--groundtruth_path', type=str, default=None)
    parser.add_argument('-id', '--image_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)