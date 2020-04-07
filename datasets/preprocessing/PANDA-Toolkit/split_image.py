
import os
import sys

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit

"""
  What is the folder structure?
"""

def main(args):

    image_root = args.data_dir
    anno_path = args.groudtruth_path
    anno_mode = args.anno_mode
    output_dir = args.output_dir
    output_anno_path = os.path.join(output_dir, 'split.json')

    os.makedirs(output_dir, exist_ok=True)

    example = PANDA_IMAGE(image_root, anno_path, annomode=anno_mode)

    splitter = ImgSplit(image_root, anno_path, anno_mode, output_dir, output_anno_path)

    splitter.splitdata(0.5)

    print('Done.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None, help='')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('-am', '--anno_mode', type=str, default='person', help='')
    parser.add_argument('-gt', '--groudtruth_path', type=str, default=None, help='')
    args = parser.parse_args()
    main(args)