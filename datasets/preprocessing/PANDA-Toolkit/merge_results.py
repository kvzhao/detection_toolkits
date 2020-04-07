
import os
import sys

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit

"""
  What is the folder structure?
"""

def main(args):
    pass

    print('Done.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None, help='')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('-gt', '--groudtruth_path', type=str, default=None, help='')
    parser.add_argument('-am', '--anno_mode', type=str, default='person', help='')
    args = parser.parse_args()
    main(args)