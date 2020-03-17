import os
import json
from os.path import join

import mmcv
import numpy as np
from tqdm import tqdm

import pycocotools.coco as coco

def main(args):

    cocoGt = coco.COCO(args.annotation_path)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


    for image_id in tqdm(cocoGt.getImgIds()):

        img_file_name = cocoGt.loadImgs(ids=[image_id])[0]['file_name']

        img_path = join(args.image_dir, img_file_name)

        #frame = cv2.imread(img_path)

        annotations = cocoGt.loadAnns(cocoGt.getAnnIds(image_id))

        gt_bboxes = []
        gt_labels = []

        for anno in annotations:
            bbox = anno['bbox']
            x, y, w, h = bbox
            category_id = anno.get('category_id', 1)
            gt_bboxes.append(
                [x, y, x + w, y + h]
            )
            gt_labels.append(category_id)

        gt_labels = np.asarray(gt_labels)
        gt_bboxes = np.asarray(gt_bboxes)
        mmcv.imshow_det_bboxes(
            join(args.image_dir, img_file_name),
            gt_bboxes,
            gt_labels,
            show=False,
            out_file=join(args.output_dir, img_file_name))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotation_path', type=str,
        default='/home/kv_zhao/datasets/CrowdHuman/annotations/val.json',
        help='Path to the odgt file')
    parser.add_argument('-id', '--image_dir', type=str,
        default='/home/kv_zhao/datasets/CrowdHuman/Images',
        help='Path to the image folder')
    parser.add_argument('-od', '--output_dir', type=str,
        default=None,
        help='Path to the coco json file')
    args = parser.parse_args()
    main(args)