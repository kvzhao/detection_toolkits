import os
import sys
from os.path import join, basename

import pandas as pd
import numpy as np
import random
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


# TODO: Use predictions rather than detres!
def merge_coco_deres():
    pass

# TODO: Use predictions rather than detres!
def merge_coco_predictions(
    src_anno_path,
    split_anno_path,
    prediction_path,
    is_nms=True,
    output_dir=None,
    nms_thresh=0.5,
    bbox_aspect_thresh=4.0):

    print('Loading source annotation json file: {}'.format(src_anno_path))

    #splitGt = COCO(split_anno_path)
    predDt = COCO(prediction_path)

    # Source Annotation
    with open(src_anno_path, 'r') as load_f:
        srcanno = json.load(load_f)

    mergedresults = defaultdict(list)
    mergedinfos = defaultdict(dict)

    for image_id in predDt.getImgIds():

        imageInfo = predDt.loadImgs(image_id)[0]
        annoInfo = predDt.loadAnns(predDt.getAnnIds(image_id))

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
            keep = py_cpu_nms(np.array(objlist), nms_thresh)
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
            aspect_ratio = h / (w + 1e-6)
            if aspect_ratio > bbox_aspect_thresh:
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

    return json_dict