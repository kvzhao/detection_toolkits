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

    num_ignore_images = 0
    num_ignore_annos = 0

    for img_id in gt.getImgIds():
        img_info = gt.loadImgs(img_id)[0]
        anno_info = gt.loadAnns(gt.getAnnIds(img_id))

        img_file_name = img_info['file_name']
        img_file_path = os.path.join(args.image_dir, img_file_name)

        if not os.path.exists(img_file_path):
            print(img_file_path, 'Not exists')
            num_ignore_images += 1
            continue

        if len(anno_info) == 0:
            num_ignore_annos += 1
            continue

        json_dict['images'].append(img_info)
        json_dict['annotations'].extend(anno_info)

    with open(args.output_file, 'w') as fp:
        json_str = json.dumps(json_dict)
        fp.write(json_str)

    print('Remove {} annotations without images, and {} empty annotations'.format(
        num_ignore_images, num_ignore_annos))
    print('Done, save to', args.output_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--input_file', type=str, default=None)
    parser.add_argument('-id', '--image_dir', type=str, default=None)
    parser.add_argument('-o', '--output_file', type=str, default=None)
    args = parser.parse_args()
    main(args)