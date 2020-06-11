import os

from os.path import basename

import json

import pycocotools.coco as coco

from utils import get_files

def main(args):
    gt = coco.COCO(args.input_file)

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': gt.loadCats(gt.getCatIds()),
    }
    image_files = [basename(f) for f in get_files(args.image_dir)]

    for img_id in gt.getImgIds():
        img_info = gt.loadImgs(img_id)[0]

        if img_info['file_name'] not in image_files:
            print(img_info, 'Not Exists, Skip')
            continue

        json_dict['images'].append(img_info)
        anno_info = gt.loadAnns(gt.getAnnIds(img_id))
        json_dict['annotations'].extend(json_dict)

    with open(args.output_file, 'w') as fp:
        json_str = json.dumps(json_dict)
        fp.write(json_str)
    print('Done, save to', args.output_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--input_file', type=str, default=None)
    parser.add_argument('-id', '--image_dir', type=str, default=None)
    parser.add_argument('-o', '--output_file', type=str, default=None)
    args = parser.parse_args()
    main(args)