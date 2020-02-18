
import os
import sys
import json
import cv2

from tqdm import tqdm

from os.path import join

# Meaning of the label should be visualized
label_map = {
    1: 'pedestrians',
    2: 'riders',
    3: 'partially-visible persons',
    4: 'ignore regions',
    5: 'crow',
}

agnostic_label_map = {
    1: 'person',
}

def read_txt2list(filename):
    with open(filename, 'r') as fp:
        content = fp.readlines()
    return [x.strip() for x in content]

def preprocess_data(data_root, filename):
    filenames = read_txt2list(join(data_root, filename))
    filenames = [f + '.jpg' for f in filenames]
    image_path = join(data_root, 'Images')
    image_paths = [join(image_path, f) for f in filenames]

    data_list = [{
        'name': name,
        'path': path,
        } for name, path in zip(filenames, image_paths)]

    return data_list

def convert2coco(annotation_path, data_list, output_path):
    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    image_id = 1
    bbox_id = 1
    person_id = 1
    ignore = 0 
    image = {}
    annotation = {}
    print('Start processing {}...'.format(output_path))
    for data in tqdm(data_list):
        annotation_list = read_txt2list(join(annotation_path, data['name'] + '.txt'))
        img = cv2.imread(data['path'])
        height, width, _ = img.shape
        image_info = {
            'file_name': data['name'],
            'height': height,
            'width': width,
            'id': image_id,
        }
        json_dict['images'].append(image_info)
        n_boxes = int(annotation_list.pop(0))
        gt_boxes, gt_classes = [], []
        for anno in annotation_list:
            cls, box = anno.split(' ')[0], anno.split(' ')[1:]
            # box: [x1, y1, x2, y2]
            box = [int(v) for v in box]
            cls = int(cls)
            gt_classes.append(cls)
            gt_boxes.append(box)
        # Parse boxes
        if n_boxes != len(gt_boxes):
            print('WARNING: n_box:{}!=gt_boxes:{}'.format(n_boxes, len(gt_boxes)))
        for class_id, bbox in zip(gt_classes, gt_boxes):
            xmin, ymin, xmax, ymax = bbox
            w = int(abs(xmax - xmin))
            h = int(abs(ymax - ymin))
            """Parse class id
                pedestrain 64.8%
                rider 0.8%
                partially visible 29.9%
                crowd 3.4%
                ignore 1.4%
            """
            annotation = {
                'area': w * h,
                'iscrowd': ignore,
                'image_id': image_id,
                'bbox': [xmin, ymin, w, h],
                'category_id': person_id,
                'id': bbox_id,
                'ignore': ignore,
                'segmentation': [],
                'finegrained_class': label_map[class_id],
            }
            json_dict['annotations'].append(annotation)
            bbox_id += 1
        image_id += 1
    for label_id, label_name in agnostic_label_map.items():
        json_dict['categories'].append({
            'supercategory': 'human',
            'id': label_id,
            'name': label_name,
        })
    with open(output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(output_path))


def main(args):

    data_root = args.data_root
    image_path = join(data_root, 'Images')
    annotation_path = join(data_root, 'Annotations')

    train_data_list = preprocess_data(data_root, 'train.txt')
    val_data_list = preprocess_data(data_root, 'val.txt')

    convert2coco(annotation_path, train_data_list, '{}_train_coco.json'.format(args.output_name))
    convert2coco(annotation_path, val_data_list, '{}_val_coco.json'.format(args.output_name))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default=None, help='Path to the dataset root')
    parser.add_argument('-o', '--output_name', type=str, default='widerperson', help='Name of the output json file.')
    args = parser.parse_args()
    main(args)
