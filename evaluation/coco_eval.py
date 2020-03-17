
import os
import sys
import json

from os.path import join
from os.path import basename

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main(args):

    cocoGt = COCO(args.groundtruth_jsonfile_path)
    cocoDt = cocoGt.loadRes(args.detection_jsonfile_path)

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.maxDets = [10, 100, 1000]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-dt', '--detection_jsonfile_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
