import os
import json
from os.path import join
import shutil
import random
import operator
import collections
import pandas as pd 
from pycocotools.coco import COCO


def main(args):
    coco = COCO(args.groundtruth_path)

    imgIds = coco.getImgIds()

    per_img_annos = {}
    for img_id in imgIds:
        annos = coco.loadAnns(coco.getAnnIds(img_id))
        num_ped = 0
        for anno in annos:
            if anno['category_id'] == 1:
                num_ped += 1
        per_img_annos[img_id] = num_ped

    per_img_annos = collections.OrderedDict(
        sorted(per_img_annos.items(), key=operator.itemgetter(1), reverse=True))

    df = pd.DataFrame(list(per_img_annos.items()), columns=['img_id', 'num'])


    people_1_3 = df[df.num.between(1, 3)]
    people_3_5 = df[df.num.between(3, 5)]
    people_5_10 = df[df.num.between(5, 10)]
    people_10_20 = df[df.num.between(10, 20)]

    print('1-3: {}, 3-5: {}, 5-10: {}, 10-20: {}'.format(
        len(people_1_3),
        len(people_3_5),
        len(people_5_10),
        len(people_10_20),
    ))

    sampled_people_1_3 = people_1_3.sample(frac=0.25)
    sampled_people_3_5 = people_3_5.sample(frac=0.5)
    sampled_people_5_10 = people_5_10.sample(frac=0.7)
    sampled_people_10_20 = people_10_20.sample(frac=0.7)

    print('1-3: {}, 3-5: {}, 5-10: {}, 10-20: {}'.format(
        len(sampled_people_1_3),
        len(sampled_people_3_5),
        len(sampled_people_5_10),
        len(sampled_people_10_20),
    ))

    sampled = pd.concat([sampled_people_1_3, sampled_people_3_5, sampled_people_5_10, sampled_people_10_20])
    print(len(sampled))

    sampled_img_ids = sampled.img_id.values
    print('sampled annos: {}'.format(sum(sampled.num.values)))

    imgInfo = coco.loadImgs(sampled_img_ids)
    print(len(imgInfo))

    annoInfo = []
    for anno in coco.loadAnns(coco.getAnnIds(sampled_img_ids)):
        if anno['category_id'] == 1:
            annoInfo.append(anno)

    print(len(annoInfo))

    catInfo = coco.loadCats(1)
    print(catInfo)

    json_dict = {
        'images': imgInfo,
        'annotations': annoInfo,
        'categories': catInfo,
    }

    with open(args.output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(args.output_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_path', type=str, default=None)
    parser.add_argument('-od', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
