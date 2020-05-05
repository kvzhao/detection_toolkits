import os
import numpy as np
import panda_utils as util
import json
from collections import defaultdict

from pycocotools.coco import COCO

CATID = {
    1: 'vbox',
    2: 'fbox',
    3: 'hbox',
}

def tlbr2tlwh(rect):
    xmin, ymin, xmax, ymax = rect
    w, h = xmax - xmin, ymax - ymin
    return [xmin, ymin, w, h]

def recttransfer(rect, scale, left, up):
    xmin, ymin, w, h = rect
    xmax, ymax = xmin + w, ymin + h
    return [int(temp / scale) for temp in [xmin + left, ymin + up, xmax + left, ymax + up]]

def py_cpu_nms(dets, thresh):
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def main(args):

    # TODO: Use coco annotation rather than original
    srcannopath = args.src_annotation_path
    split_annotation_path = args.split_annotation_path
    detection_result_path = args.detection_result_path

    is_nms = args.use_nms
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print('Loading source annotation json file: {}'.format(srcannopath))

    coco = COCO(split_annotation_path)

    mergedresults = defaultdict(list)
    mergedinfos = defaultdict(dict)

    # Source Annotation
    with open(srcannopath, 'r') as load_f:
        srcanno = json.load(load_f)

    with open(detection_result_path, 'r') as load_f:
        detresults = json.load(load_f)

    detIndexed = defaultdict(list)
    for det in detresults:
        detIndexed[det['image_id']].append(
            {'bbox': det['bbox'],
            'category_id': det['category_id'],
            'score': det['score']
            }
        )

    for image_id, annoInfo in detIndexed.items():

        imageInfo = coco.loadImgs(image_id)[0]
        #annoInfo = coco.loadAnns(coco.getAnnIds(image_id))

        filename = imageInfo['file_name']

        srcfile, paras = filename.split('___')
        srcfile = srcfile.replace('_IMG', '/IMG') + '.jpg'
        scale, left, up = paras.replace('.jpg', '').split('__')

        srcimageid = srcanno[srcfile]['image id']

        if srcimageid not in mergedinfos:
            mergedinfos[srcimageid].update(srcanno[srcfile]['image size'])
            mergedinfos[srcimageid].update({'file_name': srcfile, 'id': srcimageid})

        for objdict in annoInfo:
            #score = objdict.get('score', 1.0)
            mergedresults[srcimageid].append(
                [*recttransfer(objdict['bbox'], float(scale), int(left), int(up)),
                      objdict['score'], objdict['category_id']])

    if is_nms:
        for (imageid, objlist) in mergedresults.items():
            keep = py_cpu_nms(np.array(objlist), args.nms_thresh)
            outdets = []
            for index in keep:
                outdets.append(objlist[index])
            mergedresults[imageid] = outdets

    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [],
    }

    catset = set()
    savelist = []
    bbox_id = 0
    for (imageid, objlist) in mergedresults.items():
        json_dict['images'].append(
            mergedinfos[imageid]
        )
        for obj in objlist:
            bbox = tlbr2tlwh(obj[:4])
            # Ignore aspect ratio exceed ?
            _, _, w, h = bbox
            aspect_ratio = w / (h + 1e-6)
            if aspect_ratio > args.bbox_aspect_thresh:
                continue
            det = {
                "image_id": imageid,
                "category_id": obj[5],
                "bbox": bbox,
                "score": obj[4]
            }
            savelist.append(det)
            det.update({
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'ignore': 0,
                'segmentation': [],
                'id': bbox_id,
            })
            json_dict['annotations'].append(det)
            bbox_id += 1
            catset.add(obj[5])
    for cat in catset:
        json_dict['categories'].append({
            'supercategory': 'person',
            'id': cat,
            'name': CATID[cat]
        })

    output_path = os.path.join(output_dir, 'predictions.json')
    with open(output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(output_path))

    output_path = os.path.join(output_dir, 'detres.json')
    with open(output_path, 'w') as json_fp:
        json_str = json.dumps(savelist)
        json_fp.write(json_str)
    print('Done, save to {}'.format(output_path))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-sp', '--split_annotation_path', type=str, default=None, help='Input COCO file')
    parser.add_argument('-gt', '--src_annotation_path', type=str, default=None, help='PANDA Original Annotation')
    parser.add_argument('-dt', '--detection_result_path', type=str, default=None, help='Detectuin results json')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('--use_nms', action='store_true', help='')
    parser.add_argument('--nms_thresh', type=float, default=0.5)
    parser.add_argument('--bbox_aspect_thresh', type=float, default=3.0)
    args = parser.parse_args()
    main(args)