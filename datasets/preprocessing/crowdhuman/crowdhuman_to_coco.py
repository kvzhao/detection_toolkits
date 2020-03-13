import os
import json
from PIL import Image

from tqdm import tqdm

"""
CrowdHuman Annotation:
    {"ID": "273278,c70ac000e431e2c7",
     "gtboxes": [
         {
          "tag": "person",
          "hbox": [236, 723, 113, 139],
          "head_attr": {"ignore": 0, "occ": 0, "unsure": 0},
          "fbox": [152, 710, 284, 938],
          "vbox": [152, 710, 284, 647],
          "extra": {"box_id": 0, "occ": 0}
         },
         {
           "tag": "person",
           "hbox": [90, 631, 118, 137],
           "head_attr": {"ignore": 0, "occ": 0, "unsure": 0},
           "fbox": [-19, 623, 320, 936],
           "vbox": [0, 630, 249, 728],
           "extra": {"box_id": 1, "occ": 1}
         },
        {
            "tag": "mask",
            "hbox": [220, 352, 27, 20],
            "head_attr": {},
            "fbox": [220, 352, 27, 20],
            "vbox": [220, 352, 27, 20],
            "extra": {"ignore": 1}
        },
        ...
        ]
    }

  COCO Annotation:
    http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

  TODO:
    - Add class:head?
    - Add occlusion detect option?
    - Use full body or visible body
"""

def load_file(fpath):
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

def exceed_boundary(image_info, bbox):
    img_width, img_height = image_info['width'], image_info['height']
    x, y, w, h = bbox
    exceed = False
    if x <= 0 or y <= 0:
        exceed = True
    if w <= 0 or h <= 0:
        exceed = True
        # It fatal case
        print('img_id:{} get negative size box({}x{})'.format(
            image_info['id'], w, h))
    if x + w >= img_width or y + h >= img_height:
        exceed = True
    return exceed

def shrink_bbox(image_info, bbox):
    img_width, img_height = image_info['width'], image_info['height']
    x, y, width, height = bbox
    x = max(1, x)
    y = max(1, y)
    if width <= 0 or height <= 0:
        width = max(1, width)
        height = max(1, height)
    if x + width >= img_width: 
        width = min(img_width, x + width) -x -1
    if y + height >= img_height:
        height = min(img_height, y + height) -y -1
    width = min(width, img_width)
    width = max(width, 1)
    height = min(height, img_height)
    height = max(height, 1)
    return [x, y, width, height]

def crowdhuman2coco(odgt_path,
                    img_path,
                    json_path,
                    box_type,
                    skip_mask=False,
                    boundary_check=False,
                    add_head=False):
    """
      odgt_path: input path
      img_path: image folder path
      json_path: output path
    """
    # present the convertion strategy
    if skip_mask:
        print('Skip annotation if the tag is mask')
    records = load_file(odgt_path)
    json_dict = {"images":[], "annotations": [], "categories": []}
    START_B_BOX_ID = 1
    image_id = 1
    bbox_id = START_B_BOX_ID
    image = {}
    annotation = {}
    categories = {}
    record_list = len(records)

    num_annotation = 0
    num_annotation_exceed_boundary = 0
    num_annotation_resize_bbox = 0

    for i in tqdm(range(record_list)):
        file_name = records[i]['ID']+'.jpg'  #这里是字符串格式  eg.273278,600e5000db6370fb
        #image_id = int(records[i]['ID'].split(",")[0]) 这样会导致id唯一，要自己设定
        im = Image.open(img_path + '/' + file_name)
        image = {'file_name': file_name,
                 'height': im.size[1],
                 'width': im.size[0],
                 'id':image_id,
                }
        json_dict['images'].append(image)
        gt_box = records[i]['gtboxes']  
        gt_box_len = len(gt_box)

        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            # if category is mask, skip
            if skip_mask and category == 'mask':
                continue
            if category not in categories:
                new_id = len(categories) + 1
                categories[category] = new_id
                print('Add new category: {}'.format(category))
            category_id = categories[category]
            if box_type not in ['fbox', 'vbox', 'hbox']:
                box_type = 'fbox'
            bbox = gt_box[j][box_type]
            # Bounding Box Guardian
            timeout_counter = 0
            if boundary_check:
                is_exceed = True
                while is_exceed:
                    is_exceed = exceed_boundary(image, bbox)
                    if is_exceed:
                        bbox = shrink_bbox(image, bbox)
                        num_annotation_resize_bbox += 1
                        timeout_counter += 1
                    else:
                        is_exceed = False
                    num_annotation_exceed_boundary += 1
                    if timeout_counter > 10:
                        is_exceed = False
                        print('Timeout, skip')
                        continue
            ignore = 0
            if "ignore" in gt_box[j]['head_attr']:
                # ignore head
                ignore = gt_box[j]['head_attr']['ignore']
            if "ignore" in gt_box[j]['extra']:
                # ignore person
                ignore = gt_box[j]['extra']['ignore']
            #对字典 annotation 进行设值。
            """TODO @kv
              Whether seperate head and body into two class?
            """
            annotation = {
                'area': bbox[2] * bbox[3],
                'iscrowd': ignore,
                'image_id': image_id,
                'bbox': bbox,
                'fbox':gt_box[j]['fbox'],
                'hbox':gt_box[j]['hbox'],
                'vbox':gt_box[j]['vbox'],
                'category_id': category_id,
                'id': bbox_id,
                'ignore': ignore,
                'segmentation': [],
            }
            json_dict['annotations'].append(annotation)
            bbox_id += 1
            num_annotation += 1
        image_id += 1

    #下面这一步，对所有数据，只需执行一次，也就是对categories里的类别进行统计。
    for cate, cid in categories.items():
            #dict.items()返回列表list的所有列表项，形如这样的二元组list：［(key,value),(key,value),...］
            cat = {
                'supercategory': 'none',
                'id': cid,
                'name': cate,
            }
            json_dict['categories'].append(cat)

    #到此，json_dict的转化全部完成，对于其他的key，
    #因为没有用到（不访问），就不需要给他们空间，也不需要去处理，字典是按key访问的，如果自己需要就自己添加上去就行
    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict) #写json文件。
    json_fp.write(json_str)
    json_fp.close()

    # TODO: Show statistics
    print(categories)
    print('Parse {} annotations, resized/exceed = {}/{}'.format(
        num_annotation, num_annotation_resize_bbox, num_annotation_exceed_boundary))

def main(args):
   crowdhuman2coco(
       args.annotation,
       args.image,
       args.output,
       box_type=args.box_type,
       skip_mask=args.skip_mask,
       boundary_check=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotation', type=str, default=None, help='Path to the odgt file')
    parser.add_argument('-i', '--image', type=str, default=None, help='Path to the image folder')
    parser.add_argument('-o', '--output', type=str, default=None, help='Path to the coco json file')
    parser.add_argument('-bt', '--box_type', type=str, default='fbox', help='Type of bounding box')
    parser.add_argument('--skip_mask', action='store_true',
        help='Set true if the flag is given, skip when annotation is mask')
    parser.add_argument('-bc', '--boundary_check', action='store_true',
        help='Shrink bounding box if it exceeds the boundary')
    args = parser.parse_args()
    main(args)
