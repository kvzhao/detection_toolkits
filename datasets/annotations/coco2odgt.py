
import os
import sys
from os.path import join
from os.path import basename

import dftools
import json

# {"ID": "273271,c9db000d5146c15", 
# "gtboxes": [{"fbox": [72, 202, 163, 503], "tag": "person", "hbox": [171, 208, 62, 83], "extra": {"box_id": 0, "occ": 0}, "vbox": [72, 202, 163, 398], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}}, {"fbox": [199, 180, 144, 499], "tag": "person", "hbox": [268, 183, 60, 83], "extra": {"box_id": 1, "occ": 0}, "vbox": [199, 180, 144, 420], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}}, {"fbox": [310, 200, 162, 497], "tag": "person", "hbox": [363, 219, 54, 71], "extra": {"box_id": 2, "occ": 0}, "vbox": [310, 200, 162, 400], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}}, {"fbox": [417, 182, 139, 518], "tag": "person", "hbox": [455, 190, 53, 78], "extra": {"box_id": 3, "occ": 0}, "vbox": [417, 182, 139, 418], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}}, {"fbox": [429, 171, 224, 511], "tag": "person", "hbox": [537, 187, 55, 73], "extra": {"box_id": 4, "occ": 1}, "vbox": [534, 171, 113, 431], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}}, {"fbox": [543, 178, 262, 570], "tag": "person", "hbox": [602, 186, 71, 93], "extra": {"box_id": 5, "occ": 0}, "vbox": [543, 178, 257, 422], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}}]}

def main(args):
    df = dftools.from_coco(args.groundtruth_jsonfile_path,
                           args.detection_result_path)

    print(df.keys())

    odgtlist = []

    for _, row in df.iterrows():
        file_name = row['file_name']
        Id = file_name.rstrip('.jpg')
        if args.annotation_type == 'dtboxes':
            bboxes = row['dt_bboxes']
        elif args.annotation_type == 'gtboxes':
            bboxes = row['bboxes']
        scores = row.get('dt_scores', [1.0] * len(bboxes))
        bboxlist = []
        for bbox, score in zip(bboxes, scores):
            bboxlist.append({
                args.bbox_type: bbox,
                'score': score,
                'tag': 'person'
            })
        odgtlist.append(
                {
                    'ID': Id,
                    args.annotation_type: bboxlist,
                    'width': row['width'],
                    'height': row['height']
                }
        )

    with open(args.output_dir, 'w') as fp:
        for obj in odgtlist:
            #fp.write(json.dumps(obj))
            #fp.write('\n')
            line = json.dumps(obj) + '\n'
            fp.write(line)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str,
        default='/home/kv_zhao/datasets/CrowdHuman/annotation_val_coco.json')
    parser.add_argument('-dt', '--detection_result_path', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-bt', '--bbox_type', type=str, default='fbox')
    parser.add_argument('-at', '--annotation_type', type=str, default='dtboxes')
    args = parser.parse_args()
    main(args)
