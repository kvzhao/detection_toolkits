from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from PIL import Image
import tensorflow as tf

import collections
import hashlib
import io
import json
import logging
from tqdm import tqdm

from utils import dataset_util
from utils import label_map_util


def load_object_annotations(object_annotations_file):
    """Loads object annotation JSON file."""
    with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
        obj_annotations = json.load(fid)

    images = obj_annotations['images']
    category_index = label_map_util.create_category_index(
        obj_annotations['categories'])

    img_to_obj_annotation = collections.defaultdict(list)
    logging.info('Building bounding box index.')
    for annotation in obj_annotations['annotations']:
        image_id = annotation['image_id']
        img_to_obj_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in img_to_obj_annotation:
            missing_annotation_count += 1

    logging.info('%d images are missing bboxes.', missing_annotation_count)

    return img_to_obj_annotation, category_index


def load_images_info(images_info_file):
    with tf.io.gfile.GFile(images_info_file, 'r') as fid:
        info_dict = json.load(fid)
    return info_dict['images']

def exceed_boundary(image_info, bbox):
    img_width, img_height = image_info['width'], image_info['height']
    x, y, w, h = bbox
    exceed = False
    if w <= 0 or h <= 0:
        exceed = True
        # It fatal case
        print('img_id:{} get negative size box({}x{})'.format(
            image_info['id'], w, h))
    if x + w > img_width or y + h > img_height:
        exceed = True
    return exceed


def main(args):

    images = load_images_info(args.annotation_path)
    img_to_obj_annotation, category_index = (
        load_object_annotations(args.annotation_path))

    num_annotations_skipped = 0
    for image in tqdm(images):

        image_height = image['height']
        image_width = image['width']
        filename = image['file_name']
        image_id = image['id']
        full_path = os.path.join(args.image_dir, filename)
        bbox_annotation = img_to_obj_annotation[image_id]

        for object_annotations in bbox_annotation:
            (x, y, width, height) = tuple(object_annotations['bbox'])
            if x < 0 or y < 0:
                print('Image: {} with negative origin ({}, {})'.format(
                    image_id, x, y))
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                print('Negative bbox coord, skip')
                continue
            if x + width >= image_width or y + height >= image_height:
                num_annotations_skipped += 1
                print('Out-of-boundary bbox coord, skip')
                continue
            xmin = (float(x) / image_width)
            xmax = (float(x + width) / image_width)
            ymin = (float(y) / image_height)
            ymax = (float(y + height) / image_height)
            category_id = int(object_annotations['category_id'])
            category_index[category_id]['name'].encode('utf8')

    print('Finished writing, skipped %d annotations.', num_annotations_skipped)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotation_path', type=str, default=None, help='Path to the odgt file')
    parser.add_argument('-i', '--image_dir', type=str, default=None, help='Path to the image folder')
    args = parser.parse_args()
    main(args)
