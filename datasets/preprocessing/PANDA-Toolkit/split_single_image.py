"""
    Split single image and save coco format annotation.
"""

import os
import sys

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit

def restrain_between_0_1(values_list):
    return_list = []
    for value in values_list:
        if value < 0:
            new_value = 0
        elif value > 1:
            new_value = 1
        else:
            new_value = value
        return_list.append(new_value)
    return return_list

def RectDict2List(rectdict, imgwidth, imgheight, scale, mode='tlbr'):
    x1, y1, x2, y2 = restrain_between_0_1([rectdict['tl']['x'], rectdict['tl']['y'],
                                           rectdict['br']['x'], rectdict['br']['y']])
    xmin = int(x1 * imgwidth * scale)
    ymin = int(y1 * imgheight * scale)
    xmax = int(x2 * imgwidth * scale)
    ymax = int(y2 * imgheight * scale)

    if mode == 'tlbr':
        return xmin, ymin, xmax, ymax
    elif mode == 'tlwh':
        return xmin, ymin, xmax - xmin, ymax - ymin

def anno2coco(annos_dict, anno_types=['fbox']):
    CATEGORY = {
    'visible body': 1,
    'full body': 2,
    'head': 3,
    'vehicle': 4
    }

    CATMAP = {
        'fbox': 'full body',
        'vbox': 'visible body',
        'hbox': 'head',
        'vehicle': 'vehicle'
    }
    json_dict = {
    'images': [],
    'annotations': [],
    'categories': [
        {
        'supercategory': 'person',
        'id': CATEGORY[CATMAP[anno]],
        'name': CATMAP[anno],
        } for anno in anno_types
    ]}

    bbox_id = 1
    ignore = 0 

    for imagename, imagedict in annos_dict.items():
        imgwidth = imagedict['image size']['width']
        imgheight = imagedict['image size']['height']
        image_id = imagedict['image id']
        scale = 1.0
        #scale = showwidth / imgwidth
        image_info = {
            'file_name': imagename,
            'height': imgheight,
            'width': imgwidth,
            'id': image_id,
        }
        json_dict['images'].append(image_info)

        for object_dict in imagedict['objects list']:
            objcate = object_dict['category']
            if objcate == 'person':
                #personpose = object_dict['riding type'] if object_dict['pose'] == 'riding' else object_dict['pose']
                if 'fbox' in anno_types:
                    fullrect = RectDict2List(object_dict['rects']['full body'], imgwidth, imgheight, scale, 'tlwh')
                    annotation = {
                            'area': fullrect[2] * fullrect[3],
                            'iscrowd': ignore,
                            'image_id': image_id,
                            'bbox': fullrect,
                            'category_id': CATEGORY[CATMAP['fbox']],
                            'id': bbox_id,
                            'ignore': ignore,
                            'segmentation': [],
                    }
                    json_dict['annotations'].append(annotation)
                    bbox_id += 1

                if 'vbox' in anno_types:
                    visiblerect = RectDict2List(object_dict['rects']['visible body'], imgwidth, imgheight, scale, 'tlwh')
                    annotation = {
                            'area': visiblerect[2] * visiblerect[3],
                            'iscrowd': ignore,
                            'image_id': image_id,
                            'bbox': visiblerect,
                            'category_id': CATEGORY[CATMAP['vbox']],
                            'id': bbox_id,
                            'ignore': ignore,
                            'segmentation': [],
                    }
                    json_dict['annotations'].append(annotation)
                    bbox_id += 1

                if 'hbox' in anno_types:
                    headrect = RectDict2List(object_dict['rects']['head'], imgwidth, imgheight, scale, 'tlwh')
                    annotation = {
                            'area': headrect[2] * headrect[3],
                            'iscrowd': ignore,
                            'image_id': image_id,
                            'bbox': headrect,
                            'category_id': CATEGORY[CATMAP['hbox']],
                            'id': bbox_id,
                            'ignore': ignore,
                            'segmentation': [],
                    }
                    json_dict['annotations'].append(annotation)
                    bbox_id += 1
    return json_dict



def main(args):

    image_root = args.image_dir
    image_name = args.image_name
    anno_path = args.groudtruth_path
    anno_mode = args.anno_mode
    output_dir = args.output_dir

    output_anno_path = os.path.join(output_dir, 'split_coco.json')

    os.makedirs(output_dir, exist_ok=True)

    example = PANDA_IMAGE(image_root, anno_path, annomode=anno_mode)

    splitter = ImgSplit(image_root, anno_path, anno_mode, output_dir, output_anno_path)

    patch_annos = splitter.SplitSingle(image_name, 0.25)

    print('Done.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-id', '--image_dir', type=str, default=None, help='')
    parser.add_argument('-in', '--image_name', type=str, default=None, help='')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('-am', '--anno_mode', type=str, default='person', help='')
    parser.add_argument('-gt', '--groudtruth_path', type=str, default=None, help='')
    args = parser.parse_args()
    main(args)