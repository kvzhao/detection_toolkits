
import os
import sys
import json

def xyxy2xywh(_bbox):
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],]

def load_json(path):
    with open(path, 'r') as fp:
        jsonfile = json.load(fp)
    return jsonfile

def main(args):
    input_json = load_json(args.input_path)

    retDets = []
    for det in input_json:
        det['bbox'] = xyxy2xywh(det['bbox'])
        retDets.append(
            det
        )

    with open(args.output_path, 'w') as fp:
        json_str = json.dumps(retDets)
        fp.write(json_str)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default=None)
    args = parser.parse_args()
    main(args)