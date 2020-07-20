
import os
from os.path import join
import sys
import json

def load_json(path):
    with open(path, 'r') as fp:
        jsonfile = json.load(fp)
    return jsonfile



def main(args):

    detres = load_json(args.annotation_path)

    num_box = len(detres)
    detres = [det for det in detres if det['score'] > args.conf]
    print('Remove {} boxes under conf = {}'.format(
        num_box - len(detres), args.conf))

    with open(args.output_path, 'w') as fp:
        json_str = json.dumps(detres)
        fp.write(json_str)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, default=None, help='')
    parser.add_argument('-dt', '--annotation_path', type=str, default=None, help='')

    parser.add_argument('-c', '--conf', type=float, default=.005, help='')

    args = parser.parse_args()
    main(args)