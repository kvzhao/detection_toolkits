
import os
import sys

import panda_utils as util

from ResultMerge import DetResMerge

def main(args):
    util.GT2DetRes(args.split_file_path, args.result_file_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    outfile_name = 'mergetest.json'
    merge = DetResMerge(args.image_dir,
                        args.result_file_path,
                        args.split_file_path,
                        args.groundtruth_file_path,
                        args.output_dir,
                        outfile_name)
                        
    merge.mergeResults(is_nms=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-id', '--image_dir', type=str, default=None, help='')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('-st', '--split_file_path', type=str, default=None, help='')
    parser.add_argument('-gt', '--groundtruth_file_path', type=str, default=None, help='')
    parser.add_argument('-dt', '--detection_file_path', type=str, default=None, help='')
    parser.add_argument('-ot', '--result_file_path', type=str, default=None, help='')

    args = parser.parse_args()
    main(args)
