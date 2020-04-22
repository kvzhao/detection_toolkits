
import os
import sys
from os.path import join
from os.path import basename
import numpy as np

import dftools

from utils import get_files

from shutil import copyfile

def main(args):
    df_train = dftools.from_coco('/home/kv_zhao/datasets/CrowdHuman/annotations/train.json')
    df_val = dftools.from_coco('/home/kv_zhao/datasets/CrowdHuman/annotations/val.json')

    src_dir = '/home/kv_zhao/datasets/CrowdHuman/Images'
    dst_dir = '/home/kv_zhao/datasets/CrowdHuman/'

    for _, row in df_train.iterrows():
        file_name = row['file_name']
        copyfile(
            join(src_dir, file_name),
            join(dst_dir, 'labelme_train', file_name)
        )

    for _, row in df_val.iterrows():
        file_name = row['file_name']
        copyfile(
            join(src_dir, file_name),
            join(dst_dir, 'labelme_validation', file_name)
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-id', '--image_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)