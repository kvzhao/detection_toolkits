
import os
from os.path import join
import sys
import json

def load_json(path):
    with open(path, 'r') as fp:
        jsonfile = json.load(fp)
    return jsonfile



def in_stage_one(img_id):
    if img_id >= 391 and img_id <= 450:
        return True
    elif img_id >= 481 and img_id <= 510:
        return True
    else:
        return False
stage_one = [
    391,
    420,
    450,
    481,
    510,
]

def main(args):
    """
    input: all detres.json
    output: combined detres.json
    """

    vbox_detres = load_json(args.vbox_annotation_path)
    fbox_detres = load_json(args.fbox_annotation_path)
    hbox_detres = load_json(args.hbox_annotation_path)

    num_vbox = len(vbox_detres)
    vbox_detres = [det for det in vbox_detres if det['score'] > args.vbox_conf]
    print('Remove {} vbox under conf = {}'.format(
        num_vbox - len(vbox_detres), args.vbox_conf))

    num_fbox = len(fbox_detres)
    fbox_detres = [det for det in fbox_detres if det['score'] > args.fbox_conf]
    print('Remove {} fbox under conf = {}'.format(
        num_fbox - len(fbox_detres), args.fbox_conf))

    num_hbox = len(hbox_detres)
    hbox_detres = [det for det in hbox_detres if det['score'] > args.hbox_conf]
    print('Remove {} hbox under conf = {}'.format(
        num_hbox - len(hbox_detres), args.hbox_conf))

    person_detres = vbox_detres + fbox_detres + hbox_detres

    # TODO: Consistency check
    for det in person_detres:
        if 'area' in det:
            del det['area']
        if 'segmentation' in det:
            del det['segmentation']
        if 'iscrowd' in det:
            del det['iscrowd']
        if 'ignore' in det:
            del det['ignore']
        if 'id' in det:
            del det['id']

    if args.scenario_filter:
        print('Filter out stage II')
        num_person_detres = len(person_detres)
        person_detres = [det for det in person_detres if in_stage_one(det['image_id'])]
        print('Remove {}, {} annotations for stage I'.format(
            num_person_detres - len(person_detres), len(person_detres)))

    with open(args.output_path, 'w') as fp:
        json_str = json.dumps(person_detres)
        fp.write(json_str)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str, default=None, help='')
    parser.add_argument('-vbox', '--vbox_annotation_path', type=str, default=None, help='')
    parser.add_argument('-fbox', '--fbox_annotation_path', type=str, default=None, help='')
    parser.add_argument('-hbox', '--hbox_annotation_path', type=str, default=None, help='')

    parser.add_argument('-vc', '--vbox_conf', type=float, default=.05, help='')
    parser.add_argument('-fc', '--fbox_conf', type=float, default=.05, help='')
    parser.add_argument('-hc', '--hbox_conf', type=float, default=.05, help='')

    parser.add_argument('-f', '--scenario_filter', action='store_true')

    args = parser.parse_args()
    main(args)