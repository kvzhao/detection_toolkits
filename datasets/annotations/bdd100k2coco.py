import os
import sys
import json

from os.path import join
from os.path import basename

import imagesize

from tqdm import tqdm


def main(args):

    # json file header
    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [
        {
            'supercategory': 'vehicle',
            'id': args.category_id,
            'name': 'vehicle',
        }
        ],
    }

    #{'traffic sign', 'car', 'drivable area', 'truck', 'traffic light', 'lane', 'bike', 'motor', 'train', 'person', 'rider', 'bus'}

    vehicles = ['car', 'truck', 'bike', 'motor', 'bus']

    meta = json.load(open(args.input_file_path, 'r'))

    image_id = 0
    bbox_id = 0

    label_names = set()

    for content in tqdm(meta):
        img_name = content['name']
        annotations = content['labels']
        img_w, img_h = imagesize.get(join(args.image_dir, img_name))
        image_info = {
        'file_name': img_name,
        'height': img_h,
        'width': img_w,
        'id': image_id,
        }
        json_dict['images'].append(image_info)
        for anno in annotations:
            category_name = anno['category']
            if category_name not in vehicles:
                continue
            box = anno['box2d']
            x, y, xe, ye =  box['x1'], box['y1'], box['x2'], box['y2']
            w = xe - x
            h = ye - y
            annotation = {
                'area': w * h,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [x, y, w, h],
                'category_id': args.category_id,
                'id': bbox_id,
                'ignore': 0,
                'segmentation': [],
            }
            json_dict['annotations'].append(annotation)
            bbox_id += 1
        image_id += 1

    with open(args.output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(args.output_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('-id', '--image_dir', type=str, default=None)
    parser.add_argument('-cid', '--category_id', type=int, default=1)
    args = parser.parse_args()
    main(args)
