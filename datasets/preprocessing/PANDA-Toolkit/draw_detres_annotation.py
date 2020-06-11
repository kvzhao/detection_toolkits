import os
import json
from os.path import join

import mmcv
import numpy as np
from tqdm import tqdm

import pycocotools.coco as coco


labelname_map = {
    1: 'vbox',
    2: 'fbox',
    3: 'hbox',
    4: 'vehicle',
}
label_names = ['__background__', 'vbox', 'fbox', 'hbox', 'vehicle']

def main(args):

    cocoGt = coco.COCO(args.groundtruth_annotation_path)
    cocoDt = cocoGt.loadRes(args.detection_annotation_path)

    os.makedirs(args.output_dir, exist_ok=True)

    for image_id in tqdm(cocoGt.getImgIds()[::30]):

        img_file_name = cocoGt.loadImgs(ids=[image_id])[0]['file_name']
        print(img_file_name)

        img_path = join(args.image_dir, img_file_name)

        annotations = cocoDt.loadAnns(cocoDt.getAnnIds(image_id))

        if not annotations:
            print('WARNING: {} cannot find any annotations.'.format(img_file_name))
            continue

        gt_bboxes = []
        gt_labels = []

        for anno in annotations:
            bbox = anno['bbox']
            score = anno.get('score', 1.0)
            if args.visualization_threshold > score:
                continue
            x, y, w, h = bbox
            #y, x, w, h = bbox
            category_id = anno.get('category_id', 0)
            gt_bboxes.append(
                [x, y, x + w, y + h, score]
            )
            gt_labels.append(category_id)

        gt_labels = np.asarray(gt_labels)
        gt_bboxes = np.asarray(gt_bboxes)
        mmcv.imshow_det_bboxes(
            join(args.image_dir, img_file_name),
            gt_bboxes,
            gt_labels,
            class_names=label_names,
            thickness=2,
            show=False,
            out_file=join(args.output_dir, img_file_name))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_annotation_path', type=str,
        default='/home/kv_zhao/datasets/panda/annoCOCO_test_fbox/coco_anno.json', help='Path to the coco annotaion')
    parser.add_argument('-dt', '--detection_annotation_path', type=str,
        default=None, help='Path to the coco annotaion')
    parser.add_argument('-id', '--image_dir', type=str,
        default='/home/kv_zhao/datasets/panda/image_test', help='Path to the image folder')
    parser.add_argument('-od', '--output_dir', type=str,
        default=None,
        help='Path to output folder')
    parser.add_argument('-vt', '--visualization_threshold', type=float, default=0.4)
    args = parser.parse_args()
    main(args)