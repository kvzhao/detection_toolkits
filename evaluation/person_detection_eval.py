
import os
import sys
import json

from os.path import join
from os.path import basename

import numpy as np

import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def min_bbox_size(annos):
    min_area = 1000000
    for anno in annos:
        if anno['area'] < min_area:
            min_area = anno['area']
    return min_area


def counter(Gt, Dt):
    # temp function
    for img_id in Gt.getImgIds():
        gt = Gt.loadAnns(Gt.getAnnIds(img_id))
        dt = Dt.loadAnns(Dt.getAnnIds(img_id))
        num_gt = len(gt)
        num_dt = len(dt)
        print(img_id, num_gt - num_dt, num_gt)

        print(np.sqrt(min_bbox_size(gt)))



def main(args):

    cocoGt = COCO(args.groundtruth_jsonfile_path)
    cocoDt = cocoGt.loadRes(args.detection_jsonfile_path)

    counter(cocoGt, cocoDt)

    #print(cocoGt.getAnnIds(cocoGt.getImgIds()))

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # cocoEval.params.imgIds = 1
    cocoEval.params.maxDets = [10, 100, 1000]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    #print(cocoEval.eval)

    for eval_img in cocoEval.evalImgs:
        print(type(eval_img))
        print(eval_img['image_id'], eval_img['aRng'])
        #print(eval_img['dtMatches'], eval_img['gtMatches'])
        gtm = eval_img['gtMatches'][0]
        dtm = eval_img['dtMatches'][0]
        #for dtm in eval_img['dtMatches']:
        #    print(len(dtm))
        print('#det = {}, P = {}'.format(
            len(dtm), np.sum(dtm != 0.0) / len(dtm)))
        print('#gt = {}, R = {}'.format(
            len(gtm), np.sum(dtm != 0.0) / len(gtm)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-dt', '--detection_jsonfile_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)