import tqdm
import os
from os.path import join
from os import listdir
import numpy as np
import cv2
import json

'''
python .\mot20det.py -d C:\works -o ./
[!]One img seems corrupt : sur0395.jpg
[*]In this dataset there is not different between walker and rider.
'''


agnostic_label_map = {
    1: 'person',
}
person_id = 1
def read_file(annpath,txtname,bboxes):
    ignore_ = 'ignore' in txtname
    f = open(join(annpath,txtname),'r')
    lines = f.read().splitlines()
    for i,line in enumerate(lines):
        line_ = line.split(' ')
        
        fname = line_[0]
        if fname not in bboxes.keys():
            bboxes[fname] = []

        elemnets = [int(x) for x in line_[1:]]
        for x in range(len(elemnets)//4):
            xmin,ymin,w,h = elemnets[x*4:x*4+4]
            bboxes[fname].append((xmin,ymin,w,h,ignore_))

            
    return bboxes

def gen_coco_json(boxes,img_path):
    json_dict = {
                'images': [],
                'annotations': [],
                'categories': [],
            }
    bb_id = 1
    for image_id,image_name in enumerate(tqdm.tqdm((boxes.keys()))):
        if 'val_data' in img_path:
            p = join(img_path,image_name)
        elif 'sur' in image_name:
            p = join(img_path[0],image_name)
        elif 'ad' in image_name:
            for folder in ['ad_01','ad_02','ad_03']:
                if(os.path.exists(join(img_path[1],folder,image_name))):
                    break
            p = join(img_path[1],folder,image_name)
        
        
        with open(p, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print('Not complete image',p)
            continue

        img = cv2.imread(p)


        height, width, _ = img.shape
        image_info = {
                'file_name': image_name,
                'height': height,
                'width': width,
                'id': image_id,
        }
        json_dict['images'].append(image_info)

        for i,v in enumerate(boxes[image_name]):
            xmin,ymin,w,h,ignore = v
            annotation = {
                    'id': bb_id,
                    'image_id': image_id,
                    'category_id': person_id,
                    'segmentation': [],
                    'area': w * h,
                    'bbox': [xmin, ymin, w, h],
                    'iscrowd': 0,
                    'ignore': 1 if ignore else 0,
                }
            json_dict['annotations'].append(annotation)    
            bb_id = bb_id +1

    for label_id, label_name in agnostic_label_map.items():
        json_dict['categories'].append({
            'supercategory': 'human',
            'id': label_id,
            'name': label_name,
        })
    return json_dict
        
def write_json(args, json_dict, subset = 'train',ignore = False):
    out_ = join(args.output_path,'wp2019_{}{}.json'.format(subset,'_ignore' if ignore else ''))
    with open(out_, 'w') as json_fp:
                json_str = json.dumps(json_dict)
                print("[!]Write to ",out_)
                json_fp.write(json_str)

def main(args):
    ann_path = join(args.data_root,'WIDER Pedestrian 2019\\trainval\Annotations')
    sur_path = join(args.data_root,'WIDER Pedestrian 2019\\trainval\sur_train')
    ad_path = join(args.data_root,'WIDER Pedestrian 2019\\trainval\\ad_train')
    val_path = join(args.data_root,'WIDER Pedestrian 2019\\trainval\\val_data')
    
    
    '''
       Go through bbox.txt and ignore.txt  then write to the json.
    '''
    for p in listdir(ann_path):
        if ('list' not in p):
            bboxes = {}
            if ('train_' in p):
                print('[!]Processing :',ann_path,p)
                bboxes = read_file(ann_path,p,bboxes)
                json_dict = gen_coco_json(bboxes,[sur_path,ad_path])
                write_json(args,json_dict,subset='train',ignore='_ignore'in p)
            if ('val_' in p):
                print('[!]Processing :',ann_path,p)
                bboxes = read_file(ann_path,p,bboxes)
                json_dict = gen_coco_json(bboxes,val_path)
                write_json(args,json_dict,subset='val',ignore='_ignore'in p)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default='./', help='Path to the dataset root')
    parser.add_argument('-o', '--output_path', type=str, default='./', help='Path to the output json file.')
    args = parser.parse_args()
    main(args)
   
    
   




