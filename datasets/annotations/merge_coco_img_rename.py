
import json
import os
import sys
from os.path import join

from shutil import copyfile
from pycocotools.coco import COCO


def main(args):
    # append
    gt1 = COCO(args.annotation_path_1)
    gt2 = COCO(args.annotation_path_2)

    img_dir1 = args.source_image_dir_1
    img_dir2 = args.source_image_dir_2

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': gt1.loadCats(gt1.getCatIds()),
    }

    num_img_gt1 = len(gt1.getImgIds())
    num_img_gt2 = len(gt2.getImgIds())
    num_total_img = num_img_gt2 + num_img_gt1

    print(num_img_gt1, num_img_gt2)

    # create a new name map
    image_id_map = {}
    dst_img_id = 0
    for img_id in gt1.getImgIds():
        image_id_map['1-{}'.format(img_id)] = dst_img_id
        dst_img_id += 1
    for img_id in gt2.getImgIds():
        image_id_map['2-{}'.format(img_id)] = dst_img_id
        dst_img_id += 1

    os.makedirs(args.output_dir, exist_ok=True)
    output_img_dir = join(args.output_dir, 'images')
    output_ann_dir = join(args.output_dir, 'annotation')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)


    dst_anno_id = 0

    for img_info in gt1.loadImgs(gt1.getImgIds()):
        anns = gt1.loadAnns(gt1.getAnnIds(img_info['id']))
        dst_img_id = image_id_map['1-{}'.format(img_info['id'])]
        src_img_file_path = join(img_dir1, img_info['file_name'])
        dst_img_file_name =  str(dst_img_id).zfill(8) + '.jpg'
        dst_img_file_path = join(output_img_dir, dst_img_file_name)

        copyfile(src_img_file_path, dst_img_file_path)

        json_dict['images'].append({
            'file_name': dst_img_file_name,
            'id': dst_img_id,
            'width': img_info['width'],
            'height': img_info['height']
        })

        for anno in gt1.loadAnns(gt1.getAnnIds(img_info['id'])):
            anno['image_id'] = dst_img_id
            anno['id'] = dst_anno_id
            json_dict['annotations'].append(anno)
            dst_anno_id += 1

    for img_info in gt2.loadImgs(gt2.getImgIds()):
        anns = gt2.loadAnns(gt2.getAnnIds(img_info['id']))

        dst_img_id = image_id_map['2-{}'.format(img_info['id'])]
        src_img_file_path = join(img_dir2, img_info['file_name'])
        dst_img_file_name =  str(dst_img_id).zfill(8) + '.jpg'
        dst_img_file_path = join(output_img_dir, dst_img_file_name)

        copyfile(src_img_file_path, dst_img_file_path)

        json_dict['images'].append({
            'file_name': dst_img_file_name,
            'id': dst_img_id,
            'width': img_info['width'],
            'height': img_info['height']
        })

        for anno in gt2.loadAnns(gt2.getAnnIds(img_info['id'])):
            anno['image_id'] = dst_img_id
            anno['id'] = dst_anno_id
            json_dict['annotations'].append(anno)
            dst_anno_id += 1


    with open(join(output_ann_dir, 'coco_anno.json'), 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(args.output_dir))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt1', '--annotation_path_1', type=str, default=None)
    parser.add_argument('-gt2', '--annotation_path_2', type=str, default=None)
    parser.add_argument('-id1', '--source_image_dir_1', type=str, default=None)
    parser.add_argument('-id2', '--source_image_dir_2', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)