
import json
import os
import sys

from pycocotools.coco import COCO

def main(args):
    # append
    srcGt = COCO(args.source_annotation_path)
    dstGt = COCO(args.destination_annotation_path)
    src_imgdir = args.source_image_dir
    if src_imgdir:
        src_imgdir_name = os.path.basename(src_imgdir)
    else:
        src_imgdir_name = ''
    dst_imgdir = args.destination_image_dir
    if dst_imgdir:
        dst_imgdir_name = os.path.basename(dst_imgdir)
    else:
        dst_imgdir_name = ''

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [
            {'supercategory': 'person', 'id': 1, 'name': 'person'}
        ],
    }

    srcids = srcGt.getImgIds()
    srcannids = srcGt.getAnnIds(srcGt.getImgIds())
    srcImgInfo = srcGt.loadImgs(srcGt.getImgIds())
    srcAnnInfo = srcGt.loadAnns(srcGt.getAnnIds(srcGt.getImgIds()))
    srcCatInfo = srcGt.loadCats(srcGt.getCatIds())
    dstCatInfo = dstGt.loadCats(dstGt.getCatIds())
    # TODO check both cat are same

    print(len(srcImgInfo), len(srcAnnInfo))
    print(type(srcImgInfo), type(srcAnnInfo))

    last_src_image_id = max(srcids)
    last_src_ann_id = max(srcannids)

    print(last_src_image_id, last_src_ann_id)

    # add folder name to file_name
    if src_imgdir_name:
        for imginfo in srcImgInfo:
            imginfo['file_name'] = os.path.join(
                src_imgdir_name, imginfo['file_name'])

    json_dict['images'].extend(srcImgInfo)
    json_dict['annotations'].extend(srcAnnInfo)

    # Append new
    new_image_id = last_src_image_id + 1
    new_ann_id = last_src_ann_id + 1
    for dst_img in dstGt.loadImgs(dstGt.getImgIds()):
        dst_anns = dstGt.loadAnns(dstGt.getAnnIds(dst_img['id']))
        if dst_imgdir_name:
            file_name = os.path.join(dst_imgdir_name, dst_img['file_name'])
        else:
            file_name = dst_img['file_name']
        json_dict['images'].append(
            {
                'file_name': file_name,
                'height': dst_img['height'],
                'width': dst_img['width'],
                'id': new_image_id,
            }
        )
        for ann in dst_anns:
            json_dict['annotations'].append(
                {
                    'id': new_ann_id,
                    'image_id': new_image_id,
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd'],
                    'ignore': ann['ignore'],
                    'segmentation': ann['segmentation']
                }
            )
            new_ann_id += 1
        new_image_id += 1


    with open(args.output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(args.output_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source_annotation_path', type=str, default=None)
    parser.add_argument('-srcid', '--source_image_dir', type=str, default=None)
    parser.add_argument('-dst', '--destination_annotation_path', type=str, default=None)
    parser.add_argument('-dstid', '--destination_image_dir', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)