
#import _init_paths

import os
import cv2
import json

from os import listdir
from os.path import join
from os.path import basename
from os.path import isfile


def main(args):
    python = 'python'
    program = './box_ensemble.py'

    os.makedirs(args.output_dir, exist_ok=True)
    
    for det_dir in listdir(args.result_dirs[0]):
        output_path = join(args.output_dir, det_dir)
        os.makedirs(output_path, exist_ok=True)
        command = ' '.join([
            python, program, #task,
            '-iij', join(args.result_dirs[0],det_dir,'predictions.json'),
            '-rf', ' '.join([join(x,det_dir,'detres.json') for x in args.result_dirs]),
            '-o',join(output_path,'detres.json'),
            '-cid',str(args.category_id)
        ])
        print('[RUN]', command)
        os.system(command)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-rd','--result_dirs', nargs='+', required=True, help='')
    parser.add_argument('-od', '--output_dir', type=str, default='./pandaMerged_HG', help='The json for seeking the image_size.')
    parser.add_argument('-cid','--category_id',type=int,default=2)
    

    args = parser.parse_args()
    
    main(args)