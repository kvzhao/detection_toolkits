
import os
import sys
import json

from os.path import join
from os.path import basename

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_annotation_to_detection(Gt, json_path):
    """
    """
    resDet = []
    imgIds = Gt.getImgIds()
    Dt = COCO(json_path)
    for img_id in imgIds:
        annIds = Dt.getAnnIds(img_id)
        annotations = Dt.loadAnns(annIds)
        for anno in annotations:
            resDet.append(
                [
                    img_id, *anno['bbox'], anno['score'], anno['category_id']
                ]
            )
    resDet = np.asarray(resDet)
    resDetAnn = Gt.loadNumpyAnnotations(resDet)
    temp_output_json_path = '.tmp.json'
    with open(temp_output_json_path, 'w') as json_fp:
        json_str = json.dumps(resDetAnn)
        json_fp.write(json_str)
    print('Save detected results to {}'.format(temp_output_json_path))
    return temp_output_json_path


def main(args):

    cocoGt = COCO(args.groundtruth_jsonfile_path)
    det_path = convert_annotation_to_detection(cocoGt, args.detection_jsonfile_path)
    cocoDt = cocoGt.loadRes(det_path)

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = cocoGt.getImgIds()
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