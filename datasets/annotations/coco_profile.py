
import os
import sys
from os.path import join
from os.path import basename

import dftools


def main(args):
    df = dftools.from_coco(args.groundtruth_jsonfile_path,
                           args.detection_jsonfile_path)

    print(df)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-dt', '--detection_jsonfile_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)