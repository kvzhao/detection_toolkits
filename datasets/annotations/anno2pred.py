
import os
import sys
import json
from os.path import join
from os.path import basename

from pycocotools.coco import COCO

def main(args):
    coco = COCO(args.data_dir)
    valid_ids = [1]

    rets = []
    for img_id in coco.getImgIds():
        annos = coco.loadAnns(coco.getAnnIds(img_id))
        for anno in annos:
            bbox = anno['bbox']
            score = anno.get('score', 1.0)
            category_id = anno.get('category_id', 1)
            det = {
                'image_id': img_id,
                'bbox': bbox,
                'category_id': category_id,
                'score': score,
                }
            rets.append(det)

    output_json_path = join(args.output_dir)
    with open(output_json_path, 'w') as json_fp:
        json_str = json.dumps(rets)
        json_fp.write(json_str)
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)
