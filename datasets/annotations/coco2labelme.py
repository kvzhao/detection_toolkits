
import os
import sys
from os.path import join
from os.path import basename

import dftools


def main(args):
    df = dftools.from_coco(args.groundtruth_jsonfile_path)
    print(df)
    dftools.dump_labelme(df, args.output_dir)
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)