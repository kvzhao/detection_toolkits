import os
from os.path import join
import sys
import json
from pprint import pprint

from collections import defaultdict


from pycocotools.coco import COCO

def load_json(path):
    with open(path, 'r') as fp:
        jsonfile = json.load(fp)
    return jsonfile

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

def main(args):
    output_dir = args.output_dir
    panda_annotation_path = args.panda_annotation_path

    anno_types = args.selected_annotation_types

    print(anno_types)

    panda_annos = load_json(panda_annotation_path)

    os.makedirs(output_dir, exist_ok=True)

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [
            {
            'supercategory': 'person',
            'id': CATEGORY[CATMAP[anno]],
            'name': CATMAP[anno],
            } for anno in anno_types
        ],
    }

    bbox_id = 1
    ignore = 0 

    scene_ids = defaultdict(list)

    # per splitted image
    # TODO: Aggregate same large image or same scenario
    for imagename, imagedict in panda_annos.items():
        imgwidth = imagedict['image size']['width']
        imgheight = imagedict['image size']['height']
        image_id = imagedict['image id']
        scale = 1.0

        scenename = imagename.split('/IMG')[0]
        scene_ids[scenename].append(image_id)

        #scale = showwidth / imgwidth
        image_info = {
            'file_name': imagename,
            'height': imgheight,
            'width': imgwidth,
            'id': image_id,
        }
        json_dict['images'].append(image_info)

        # per annotation
        for object_dict in imagedict.get('objects list', []):
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

    output_path = join(output_dir, 'coco_anno.json')
    with open(output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(output_path))

    # seperate into scenes and single images
    coco = COCO(output_path)
    scene_output_path = join(output_dir, 'scenes')
    os.makedirs(scene_output_path, exist_ok=True)

    for name, ids in scene_ids.items():
        imgInfo = coco.loadImgs(ids)
        annoInfo = coco.loadAnns(coco.getAnnIds(ids))
        catInfo = [
            {
            'supercategory': 'person',
            'id': CATEGORY[CATMAP[anno]],
            'name': CATMAP[anno],
            } for anno in anno_types
        ]
        json_dict = {
            'images': imgInfo,
            'annotations': annoInfo,
            'categories': catInfo,
        }
        output_path = join(scene_output_path, name) + '.json'
        with open(output_path, 'w') as json_fp:
            json_str = json.dumps(json_dict)
            json_fp.write(json_str)

    print('Done.')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Convert split.json to coco annotation format.')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('-gt', '--panda_annotation_path', type=str, default=None, help='')
    parser.add_argument('-at', '--selected_annotation_types', type=str, nargs='+', default=['fbox'],
        help='Choose annotation to export: fbox, vbox, hbox, vehicle')
    args = parser.parse_args()
    main(args)