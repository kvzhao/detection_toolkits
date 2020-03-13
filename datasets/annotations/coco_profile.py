
import os
import sys
from os.path import join
from os.path import basename
import numpy as np

import dftools


def main(args):
    df = dftools.from_coco(args.groundtruth_jsonfile_path,
                           args.detection_jsonfile_path)

    df = dftools.compute_bbox_per_image(df)
    print(df)
    print(np.mean(df.num_bbox), np.min(df.num_bbox), np.max(df.num_bbox))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-dt', '--detection_jsonfile_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)