from ensemble_boxes import nms,weighted_boxes_fusion
import json
from pycocotools.coco import COCO

def image_size_source(args):
    source = COCO(args.image_id_json)
    whs = {}
    for k,image_info in source.imgs.items():
        whs[image_info['id']] = (image_info['width'],image_info['height'])
    return whs
def xywh_to_xyxy(boxlist):
    return [boxlist[0],boxlist[1],boxlist[0]+boxlist[2],boxlist[1]+boxlist[3]]
def xyxy_to_xywh(boxlist):
    return [int(boxlist[0]),int(boxlist[1]),int(boxlist[2]-boxlist[0]),int(boxlist[3]-boxlist[1])]
def normalize(boxlist,wh):
    return [boxlist[0]/wh[0],boxlist[1]/wh[1],boxlist[2]/wh[0],boxlist[3]/wh[1]]
def de_normalize(boxlist,wh):
    return [boxlist[0]*wh[0],boxlist[1]*wh[1],boxlist[2]*wh[0],boxlist[3]*wh[1]] 
def main(args):
    def get_imggroup():
        return dict(boxes_list = [],scores_list = [],labels_list = [])
    whs = image_size_source(args)
    weights = [1] * len(args.result_files)
    imggroups = {}
    for result_index,result_file in enumerate(args.result_files):    
        print('[~]',result_file)
        with open(result_file, 'r') as reader:
            jf = json.loads(reader.read())
            for box in jf:
                if box['image_id'] not in imggroups:
                    imggroups[box['image_id']] = get_imggroup()
                if len(imggroups[box['image_id']]['boxes_list']) == result_index:
                    imggroups[box['image_id']]['boxes_list'].append([])
                if len(imggroups[box['image_id']]['scores_list']) == result_index:
                    imggroups[box['image_id']]['scores_list'].append([])
                if len(imggroups[box['image_id']]['labels_list']) == result_index:
                    imggroups[box['image_id']]['labels_list'].append([])

                box_ = xywh_to_xyxy(box['bbox'])
                box_ = normalize(box_,whs[box['image_id']])
                try:
                    imggroups[box['image_id']]['boxes_list'][result_index].append(box_)
                except:
                    print(box['image_id'], result_file)
                imggroups[box['image_id']]['scores_list'][result_index].append(box['score'])
                imggroups[box['image_id']]['labels_list'][result_index].append(args.category_id)
    fusion_result = []
    for k,v in imggroups.items():
        boxes, scores, labels = weighted_boxes_fusion(v['boxes_list'], v['scores_list'], v['labels_list'], weights=weights, iou_thr=args.iou_thr, skip_box_thr=args.skip_box_thr)
        for i in range(len(boxes)):
            box = {
                'image_id' : k,
                'bbox' :[int(x) for x in xyxy_to_xywh(de_normalize(boxes[i],whs[k]))],
                'category_id' : int(labels[i]),
                'score' : float(scores[i]),
            }
            fusion_result.append(box)
    with open(args.output_path, 'w') as json_fp:
                json_str = json.dumps(fusion_result)
                json_fp.write(json_str)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rf','--result_files', nargs='+', required=True)
    
    parser.add_argument('-iij', '--image_id_json', type=str,
        default='/home/kv_zhao/datasets/panda/annoCOCO_test_fbox/coco_anno.json', help='The json for seeking the image_size.')
    parser.add_argument('-o', '--output_path', type=str, default='./merged_boxes.json', help='Path to the output json file.')

    parser.add_argument('--iou_thr', type=float, default=0.6)
    parser.add_argument('--skip_box_thr', type=float, default=0.0001)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('-cid','--category_id',type=int, default=2,
        help='ID: vbox=1, fbox=2, hbox=3')

    args = parser.parse_args()
    main(args)
