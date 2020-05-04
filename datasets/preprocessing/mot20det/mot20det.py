import tqdm
import os
from os.path import join
from os import listdir
import numpy as np


import json
import shutil
'''
python .\mot20det.py -d C:\works -o C:\works
[*]There is no ignore instance in this dataset.
'''
# map follow the https://motchallenge.net/data/MOT20/
resolution_map = {
    'train/MOT20-01':(1920,1080),
    'train/MOT20-02':(1920,1080),
    'train/MOT20-03':(1173,880),
    'test/MOT20-04':(1545,1080),
    'train/MOT20-05':(1654,1080),
    'test/MOT20-06':(1920,734),
    'test/MOT20-07':(1920,1080),
    'test/MOT20-08':(1920,734),
}



agnostic_label_map = {
    1: 'person',
}

person_id = 1
def read_file(path):
    f = open(path,"r")
    lines = f.read().splitlines()
    result = {}
    max_fnum = 0
    abset = set()

    for i,line in enumerate(lines):
        line_ = line.split(',')
        if max_fnum < int(line_[0]):
            max_fnum = int(line_[0])
        elements = list(map(lambda x: int(float(x)),line_))

        '''
        a : moving(1) or not(0)
        b : walking people(1), workers(7,6), background or negtive objects(13,11)
        '''
        _,target_id,xmin,ymin,w,h,a,b,conf = elements

        if line_[0].zfill(6)+'.jpg' not in result.keys():
            result[line_[0].zfill(6)+'.jpg'] = []
        '''
        if(b not in[13,11]):
            continue'''
        if(b not in[7,6,1]):
            continue
        abset.add((a,b))
        result[line_[0].zfill(6)+'.jpg'].append((xmin,ymin,w,h,a))
    
    return result,max_fnum

def gen_coco_json(boxes,max_fnum,folder_name):
    json_dict = {
                'images': [],
                'annotations': [],
                'categories': [],
            }
    bb_id = 1
    for fid in range(1,max_fnum+1):
        fname_str = str(fid).zfill(6)+'.jpg'
        image_info = {
                'file_name': fname_str,
                'height': resolution_map[folder_name][1],
                'width': resolution_map[folder_name][0],
                'id': fid,
        }
        json_dict['images'].append(image_info)
        for i,v in enumerate(boxes[fname_str]):
            xmin,ymin,w,h,moving = v
            annotation = {
                    'id': bb_id,
                    'image_id': fid,
                    'category_id': person_id,
                    'segmentation': [],
                    'area': w * h,
                    'bbox': [xmin, ymin, w, h],
                    'iscrowd': 0,
                    'ignore': 0,
                    'moving' : moving #Addition
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
        
def main(args):
    root = args.data_root
    path = join(root,'MOT20\MOT20DetLabels\MOT20DetLabels')
    for key in tqdm.tqdm(resolution_map.keys()):
        if 'train' in key:
            boxes,max_fnum = read_file(join(path,key,'gt','gt.txt'))
            json_dict = gen_coco_json(boxes,max_fnum,key)
            with open(join(args.output_path,key.split('/')[1]+'.json'), 'w') as json_fp:
                json_str = json.dumps(json_dict)
                json_fp.write(json_str)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default='./', help='Path to the dataset root')
    parser.add_argument('-o', '--output_path', type=str, default='./', help='Path to the output json file.')
    args = parser.parse_args()
    main(args)

