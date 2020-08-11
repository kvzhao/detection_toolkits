import os
import json
from os.path import join
import shutil
import random
import operator
import collections
import pandas as pd 
import numpy as np
from pycocotools.coco import COCO


def main(args):
    coco = COCO(args.groundtruth_path)

    imgIds = coco.getImgIds()

    per_img_stats = {}

    for img_id in imgIds:
        img_info = coco.loadImgs(img_id)[0]
        annos = coco.loadAnns(coco.getAnnIds(img_id))
        for anno in annos:
            box_size = max(anno["bbox"][2], anno["bbox"][3])
        per_img_stats[img_id] = {
            "img_size": max(img_info["width"], img_info["height"]),
            "box_size": box_size,
            "img_id": img_id
        }

    df = pd.DataFrame.from_dict(per_img_stats, orient="index")

    df.to_csv("celeba_stats.csv")
    print(df)

    for img_rng in [[0, 320], [320, 640], [640, 1080], [1080, 1920], [1920, 9000]]:
        _n = len(df[df.img_size.between(*img_rng)])
        print("Img size within [{}, {}]: {}".format(img_rng[0], img_rng[1], _n))

    sampled_df = df.sample(frac=args.sampled_number / len(df), random_state=200)
    others = df.drop(sampled_df.index)
    sampled_img_ids = list(sampled_df.img_id.values)
    other_img_ids = list(others.img_id.values)


    os.makedirs(args.output_dir, exist_ok=True)
    output_path = join(args.output_dir, "sampled_anno_coco.json")

    def save_coco_anno(coco, img_ids, path):

        imgInfo = coco.loadImgs(img_ids)
        print(len(imgInfo))

        annoInfo = []
        for anno in coco.loadAnns(coco.getAnnIds(img_ids)):
            if anno['category_id'] == 1:
                annoInfo.append(anno)

        catInfo = coco.loadCats(1)

        json_dict = {
            'images': imgInfo,
            'annotations': annoInfo,
            'categories': catInfo,
        }

        with open(path, 'w') as json_fp:
            json_str = json.dumps(json_dict)
            json_fp.write(json_str)
        print('Done, save to {}'.format(path))

    save_coco_anno(coco, sampled_img_ids, output_path)
    output_path = join(args.output_dir, "others_anno_coco.json")
    save_coco_anno(coco, other_img_ids, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_path', type=str,
        default="/home/kv_zhao/datasets/CelebA/celeba_anno_coco.json")
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-n', '--sampled_number', type=int, default=12000)
    args = parser.parse_args()
    main(args)
