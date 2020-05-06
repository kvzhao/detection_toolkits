import os
import json

from pycocotools.coco import COCO

def main(args):

    coco = COCO(args.input_path)

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': coco.loadCats(coco.getCatIds())
    }

    bbox_id = 0
    for img_info in coco.loadImgs(coco.getImgIds()):
        anns = coco.loadAnns(coco.getAnnIds(img_info['id']))
        json_dict['images'].append(
            {
                'file_name': img_info['file_name'],
                'height': img_info['height'],
                'width': img_info['width'],
                'id': img_info['id'],
            }
        )
        for ann in anns:
            if ann.get('score', 1.0) < args.conf:
                print('Skip {}'.format(ann))
                continue
            json_dict['annotations'].append(
                {
                    'id': bbox_id,
                    'image_id': img_info['id'],
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd'],
                    'ignore': ann['ignore'],
                    'segmentation': ann['segmentation']
                }
            )

    with open(args.output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(args.output_path))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default=None)
    parser.add_argument('-c', '--conf', type=float, default=0.8)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)