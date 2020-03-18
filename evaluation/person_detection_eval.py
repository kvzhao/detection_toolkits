
import os
import sys
import json

from os.path import join
from os.path import basename

import numpy as np

import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pytablewriter import MarkdownTableWriter



def PerImageSummarize(cocoEval):
    """
      - FPPI
      - MR
      - Density classification

      order [T, R, K, A, M]
        T: [.5:.05:.95] T=10 IoU thresholds for evaluatio
        R: [0:.01:1] R=101 recall thresholds for evaluation
        K: category
        A: areaRng    - [...] A=4 object area ranges for evaluation
        M: maxDets    - [1 10 100] M=3 thresholds on max detections per image
    """
    # per
    iou_05 = 0

    per_image_precision_recall = {}

    # NOTE: WARNING Hard-coded
    offset = len(cocoEval.params.catIds) + len(cocoEval.params.areaRng) - 1

    for eval_img in cocoEval.evalImgs[::offset]:
        # IoU = 0.5 index == 0
        img_id = eval_img['image_id']
        gtm = eval_img['gtMatches'][iou_05]
        dtm = eval_img['dtMatches'][iou_05]

        #print('#det = {}, P = {}'.format(
        #    len(dtm), np.sum(dtm != 0.0) / len(dtm)))
        #print('#gt = {}, R = {}'.format(
        #    len(gtm), np.sum(dtm != 0.0) / len(gtm)))
        print(np.sum(gtm != 0.0), np.sum(dtm != 0.0))
        num_tp = np.sum(dtm != 0.0)
        num_fp = np.sum(dtm == 0.0)
        num_dt = len(dtm)
        num_gt = len(gtm)
        per_image_precision_recall[img_id] = {
            'num_tp': num_tp,
            'num_fp': num_fp,
            'precision': num_tp / num_dt,
            'recall': np.sum(gtm != 0.0) / num_gt,
            'num_gt': num_gt,
            'num_dt': num_dt,
            'num_miss': num_gt - num_tp,
        }

    per_image_report = pd.DataFrame.from_dict(
        per_image_precision_recall, orient='index')

    print(per_image_report)
    print('Mean Precision: {}'.format(per_image_report.precision.mean()))
    print('Mean Recall: {}'.format(per_image_report.recall.mean()))
    print('FPPI: {}'.format(per_image_report.num_fp.sum() / len(per_image_report)))
    print('Missing Rate: {}%'.format(
        100.0 * (per_image_report.num_miss.sum() / per_image_report.num_gt.sum())))

    # TODO: Density report
    num_bbox_ranges = [[0, 10], [10, 50], [50, 100], [100, 300], [300, 1000]]

    writer = MarkdownTableWriter()
    writer.table_name = 'PR & Density'
    writer.headers = ['#person', 'Precision', 'Recall']
    values = []
    for nb in num_bbox_ranges:
        nb_str = '{} ~ {}'.format(nb[0], nb[1])
        p = per_image_report[per_image_report.num_gt.between(nb[0], nb[1])].precision.mean()
        r = per_image_report[per_image_report.num_gt.between(nb[0], nb[1])].recall.mean()
        values.append([nb_str, p, r])
    writer.value_matrix = values
    writer.margin = 1
    writer.write_table()

    return per_image_report


def main(args):

    cocoGt = COCO(args.groundtruth_jsonfile_path)
    cocoDt = cocoGt.loadRes(args.detection_jsonfile_path)


    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # cocoEval.params.imgIds = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    report = PerImageSummarize(cocoEval)

    if args.output_path is None:
        report.to_csv('per_image_report.csv')
    else:
        report.to_csv(args.output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-dt', '--detection_jsonfile_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)