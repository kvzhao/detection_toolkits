
import os
import cv2
import json

from os import listdir
from os.path import join
from os.path import basename
from os.path import isfile

def main(args):
    root_dir = args.root_dir
    split_gt_folder = args.split_groundtruth_folder
    dt_folder = args.detection_result_folder
    gt_folder = args.groundtruth_folder
    output_dir = args.output_dir
    oracle_path = args.oracle_path
    os.makedirs(output_dir, exist_ok=True)

    # TODO: remove the hard-coded
    python = 'python3'
    eval_program = join(root_dir, 'evaluation/coco_eval.py')

    for (dirpath, dirnames, filenames) in os.walk(dt_folder):
        if not filenames:
            continue
        if 'detres.json' not in filenames:
            continue
        detres_path = join(dirpath, 'detres.json')
        scenename = os.path.basename(dirpath)

        gt_path = join(gt_folder, scenename) + '.json'
        split_gt_path = join(split_gt_folder, scenename) + '.json'
        merged_detres_path = join(dt_folder, scenename, 'detres.json')

        if not os.path.exists(merged_detres_path):
            print('WARNING! {} not exist'.format(merged_detres_path))
            continue
        # Evaluate
        eval_command = ' '.join([
            python,
            eval_program,
            '-gt', gt_path,
            '-dt', merged_detres_path, 
        ])

        print('====== {} ======'.format(scenename))
        os.system(eval_command)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    
    # inference configs 
    parser.add_argument('--root_dir', type=str, default='/home/kv_zhao/detection_toolkits')
    parser.add_argument('-id', '--image_dir', type=str,
        default='/home/kv_zhao/datasets/panda/split/PANDA_IMAGE/', help='Input image directory')
    parser.add_argument('-sgt', '--split_groundtruth_folder', type=str,
        default='/home/kv_zhao/datasets/panda/split/anno_split_train/scenes/', help='Coco annotation file path.')
    parser.add_argument('--oracle_path', type=str,
        default='/home/kv_zhao/datasets/panda/annoJSONs/person_bbox_train.json', help='Coco annotation file path.')
    parser.add_argument('-gt', '--groundtruth_folder', type=str,
        default='/home/kv_zhao/datasets/panda/annoCOCO_fbox/scenes/', help='Coco annotation file path.')
    parser.add_argument('-dt', '--detection_result_folder', type=str,
        default='/home/kv_zhao/datasets/panda/split/prelabel/CD1_all_scenarios/', help='Model results folder.')
    parser.add_argument('-od', '--output_dir', type=str,
        default='/home/kv_zhao/datasets/panda/split/merged_prelabel/CD1_all_scenarios/', help='Output directory')

    args = parser.parse_args()
    
    main(args)