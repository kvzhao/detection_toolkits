
import os
import sys
import json

from os.path import join
from os.path import basename

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def CustomizedSummarize(self):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    def _summarizeDets():
        stats = np.zeros((23,))
        # Precision
        stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        # Recall
        stats[3] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[4] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[5] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])

        # Size Effect
        stats[6] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[7] = _summarize(1, iouThr=.5, areaRng='xsmall', maxDets=self.params.maxDets[2])
        stats[8] = _summarize(1, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[2])
        stats[9] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(1, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, iouThr=.5, areaRng='xsmall', maxDets=self.params.maxDets[2])
        stats[12] = _summarize(0, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[2])
        stats[13] = _summarize(0, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[14] = _summarize(0, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[2])

        stats[15] = _summarize(1, areaRng='xsmall', maxDets=self.params.maxDets[2])
        stats[16] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[17] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[18] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[19] = _summarize(0, areaRng='xsmall', maxDets=self.params.maxDets[2])
        stats[20] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[21] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[22] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats

    if not self.eval:
        raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'bbox':
        summarize = _summarizeDets
    self.stats = summarize()


def main(args):

    cocoGt = COCO(args.groundtruth_jsonfile_path)
    cocoDt = cocoGt.loadRes(args.detection_jsonfile_path)

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.maxDets = [10, 100, 500]
    thres0, thres1, thres2, thres3, thres4 = [0, 32, 200, 400, 1e5]
    cocoEval.params.areaRng = [[thres0 ** 2, thres4 ** 2],
                               [thres0 ** 2, thres1 ** 2],
                               [thres1 ** 2, thres2 ** 2],
                               [thres2 ** 2, thres3 ** 2],
                               [thres3 ** 2, thres4 ** 2],
                               ]

    cocoEval.evaluate()
    cocoEval.accumulate()
    # official outcomes
    # cocoEval.summarize()
    # customized outcomes
    CustomizedSummarize(cocoEval)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-dt', '--detection_jsonfile_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
