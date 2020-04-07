
import os
import sys

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ResultMerge import DetResMerge

"""
  What is the folder structure?
"""

def main(args):
    groundtruth_path = args.groundtruth_path
    detection_path = args.detection_path
    split_file_path = args.split_file_path

    input_dir = args.data_dir
    output_dir = args.output_dir

    merge = DetResMerge(input_dir, detection_path, split_file_path, groundtruth_path, output_dir, 'mergetest.json')

    print('Done.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None, help='')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('-gt', '--groundtruth_path', type=str, default=None, help='')
    parser.add_argument('-sp', '--split_file_path', type=str, default=None, help='')
    parser.add_argument('-dt', '--detection_path', type=str, default=None, help='')
    parser.add_argument('-am', '--anno_mode', type=str, default='person', help='')
    args = parser.parse_args()
    main(args)