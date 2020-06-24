
import os
import json
from os.path import join

import imagesize
from tqdm import tqdm

def get_files(path, ext=('.txt')):
    # ext is str or tuple
    files = []
    for (dir_path, _, file_names) in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(ext):
                files.append(join(dir_path, file_name))
    return files


def read_anno_file(filename):
    with open(filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    contents = [x.strip() for x in content] 
    annos = []

    for content in contents:
        elems = content.split(' ')
        annos.append({
            'class': int(elems[0]),
            'bbox': list(map(float, elems[1:]))
        })
    return annos


def yolo2coco(box, img_w, img_h):
    x1, y1 = int((box[0] + box[2]/2) * img_w), int((box[1] + box[3]/2) * img_h)
    x2, y2 = int((box[0] - box[2]/2) * img_w), int((box[1] - box[3]/2) * img_h)
    return [x1, y1, x2 - x1, y2 - y1]



def main(args):

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [{
            'supercategory': 'car',
            'id': 1,
            'name': 'car'
        }]
    }


    data_dir = args.data_dir
    output_path = args.output_path

    anno_files = get_files(data_dir)

    img_id = 0
    anno_id = 0

    for anno_file_path in tqdm(anno_files):

        # get image info
        anno_file_name = os.path.basename(anno_file_path).rstrip('.txt')
        image_name = anno_file_name + '.jpg'

        # (width, height)
        img_width, img_height = imagesize.get(join(data_dir, image_name))

        img_info = {
            'file_name': image_name,
            'width': img_width,
            'height': img_height,
            'id': img_id,
        }
        json_dict['images'].append(img_info)

        # get annotataions
        annos = read_anno_file(anno_file_path)

        for anno in annos:
            bbox = anno['bbox']

            # yolo to coco format
            coco_bbox = yolo2coco(bbox, img_width, img_height)

            anno_info = {
                'image_id': img_id,
                'area': coco_bbox[2] * coco_bbox[3],
                'iscrowd': 0,
                'ignore': 0,
                'bbox': coco_bbox,
                'category_id': 1,
                'id': anno_id,
                'segmentation': [],
            }
            json_dict['annotations'].append(anno_info)

            anno_id += 1
        img_id += 1

    with open(output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict) #写json文件。
        json_fp.write(json_str)
        json_fp.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
