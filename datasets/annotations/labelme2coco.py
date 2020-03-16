
import os
import sys
from os.path import join
from os.path import basename

import dftools


def main(args):
    dftools.convert_labelme_to_coco(args.data_dir, args.output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)
