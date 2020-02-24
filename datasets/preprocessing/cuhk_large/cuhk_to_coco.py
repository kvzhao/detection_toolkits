
import os
import cv2
import json
from tqdm import tqdm
from cuhk_large import CUHK_Large


def convert2coco(filenames, cuhk, name2box, outname):
    """
      Args:
        filenames: list of filenames of train or eval images
        cuhk: handler
        name2box: a dict maps name to box
      Return:
        coco_json: a json file
      NOTE:
        In CUHK, we only export agnostic label
    """
    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    image_id = 1
    bbox_id = 1
    ignore = 0 
    image = {}
    annotation = {}

    category_id = 1

    print('Start processing {}...'.format(outname))
    for name in tqdm(filenames):
        img = cv2.imread(cuhk.get_image_path(name))
        height, width, _ = img.shape

        image_info = {
            'file_name': name,
            'height': height,
            'width': width,
            'id': image_id,
        }
        json_dict['images'].append(image_info)
        gt_boxes = name2box[name]

        for bbox in gt_boxes:
            # Box: [xmin, ymin, w, h]
            xmin, ymin, w, h = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            w = int(w)
            h = int(h)
            annotation = {
                'area': w * h,
                'iscrowd': ignore,
                'image_id': image_id,
                'bbox': [xmin, ymin, w, h],
                'category_id': category_id,
                'id': bbox_id,
                'ignore': ignore,
                'segmentation': [],
            }
            json_dict['annotations'].append(annotation)
            bbox_id += 1
        image_id += 1

    json_dict['categories'].append({
        'supercategory': 'human',
        'id': category_id,
        'name': 'person',
    })

    with open(outname, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(outname))

def main(args):

    data_root = args.data_root
    if data_root is None:
        raise ValueError('data_root must be assigned')
    cuhk = CUHK_Large(data_root)

    annos = cuhk.get_image_annotation()
    print('#of annotations: {}'.format(len(annos)))

    filename2bbox = {}
    for anno in annos:
        img_name, n_box, bboxes = anno
        img_name = img_name[0]
        bboxes = [box[0][0] for box in bboxes[0]]
        if n_box != len(bboxes):
            print('WARNING: {} annotation not consistent'.format(img_name))
        filename2bbox[img_name] = bboxes

    all_image_names = filename2bbox.keys()
    test_image_names = [name[0][0] for name in cuhk.pool_annotation]
    train_image_names = list(set(all_image_names) ^ set(test_image_names))

    convert2coco(train_image_names, cuhk, filename2bbox, outname=args.output_name+'_train.json')
    convert2coco(test_image_names, cuhk, filename2bbox, outname=args.output_name+'_test.json')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default=None, help='Path to the dataset root')
    parser.add_argument('-o', '--output_name', type=str, default='cuhk_coco_format', help='Name of the output json file.')
    args = parser.parse_args()
    main(args)
