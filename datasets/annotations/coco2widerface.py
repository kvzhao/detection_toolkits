
import os
import sys
from os.path import join
from os.path import basename

import json
import math
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


"""Widerface format

Folder / image_name.txt

    image_name
    #of faces
    x, y, w, h, score
    ...
"""


def main(args):

    Gt = COCO(args.groundtruth_jsonfile_path)
    Dt = Gt.loadRes(args.detection_result_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_id in tqdm(Dt.getImgIds()):
        img_info = Dt.loadImgs(img_id)[0]

        folder_file_name = img_info['file_name']

        folder_name = folder_file_name.split('/')[0]
        file_name = basename(folder_file_name.split('/')[1]).rstrip('.jpg')

        preds = Dt.loadAnns(Dt.getAnnIds(img_id))

        bboxes = [p['bbox'] + [p['score']] for p in preds]

        output_folder = join(args.output_dir, folder_name)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        fout = open(os.path.join(output_folder, file_name + '.txt'), 'w')
        fout.write(file_name + '\n')

        fout.write(str(len(bboxes)) + '\n')
        for bbox in bboxes:
            score = bbox[4]
            if score == 0.0:
                continue
            fout.write('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3]), score if score <= 1 else 1) + '\n')
        fout.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str,
        default='/home/kv_zhao/datasets/WiderFace/widerface_val_coco.json')
    parser.add_argument('-dt', '--detection_result_path', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)
