import os
import sys
from os.path import join
from os.path import basename

import dftools


def main(args):

    if not args.annotation_path_1 or not args.annotation_path_2:
        raise ValueError()

    df1 = dftools.from_coco(args.annotation_path_1)
    df2 = dftools.from_coco(args.annotation_path_2)

    merged = dftools.merge(df1, df2)
    print(merged)
    print(len(merged))

    if args.output_path:
        dftools.dump_coco(merged, args.output_path)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a1', '--annotation_path_1', type=str, default=None)
    parser.add_argument('-a2', '--annotation_path_2', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
