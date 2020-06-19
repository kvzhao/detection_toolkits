

import os
import sys
import json
import cv2

from tqdm import tqdm

from os.path import join

import scipy.io


def get_files(path, ext=('.png', '.jpg')):
    # ext is str or tuple
    files = {}
    for (dir_path, _, file_names) in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(ext):
                files[file_name] = {
                    'path': dir_path,
                    'dir': os.path.basename(dir_path),
                }
    return files

def main(args):

    image_dir_and_name_dict = get_files(args.image_dir)

    annotations = scipy.io.loadmat(args.groundtruth_path)

    label_id = 1

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [
            {'supercategory': 'face', 'id': label_id, 'name':'face'}
        ],
    }

    image_id = 0
    bbox_id = 0
    ignore = 0 

    folders = annotations['file_list']
    annotation_list = annotations['face_bbx_list']
    occlusion_label_list = annotations['occlusion_label_list']
    invalid_label_list = annotations['invalid_label_list']
    expression_label_list = annotations['expression_label_list']
    illumination_label_list = annotations['illumination_label_list']
    event_list = annotations['event_list']
    blur_label_list = annotations['blur_label_list']
    pose_label_list = annotations['pose_label_list']
    print(len(folders), 'folders')

    for folder, annos, occs in zip(folders, annotation_list, occlusion_label_list):
        folder = folder[0]
        annos = annos[0]
        occs = occs[0]

        for img_name, anno, occ in zip(folder[:, 0], annos[:, 0], occs[:, 0]):

            img_name = '{}.jpg'.format(img_name[0])
            img_path = join(image_dir_and_name_dict[img_name]['path'], img_name)

            image = cv2.imread(img_path)

            img_h, img_w, _ = image.shape

            img_file_name = join(image_dir_and_name_dict[img_name]['dir'], img_name)

            img_info = {
                'file_name': img_file_name,
                'height': img_h,
                'width': img_w,
                'id': image_id,
            }
            json_dict['images'].append(img_info)

            for bbox, occ_id in zip(anno, occ[:, 0]):
                annotation = {
                    'area': int(bbox[2] * bbox[3]),
                    'iscrowd': ignore,
                    'image_id': image_id,
                    'bbox': bbox.tolist(),
                    'category_id': label_id,
                    'id': bbox_id,
                    'ignore': ignore,
                    'segmentation': [],
                    'occ': int(occ_id),
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
    parser.add_argument('-id', '--image_dir', type=str, default=None)
    parser.add_argument('-gt', '--groundtruth_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
