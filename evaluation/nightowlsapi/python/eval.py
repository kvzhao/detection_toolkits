import os
from os.path import join
from coco import COCO
from eval_MR_multisetup import COCOeval


def main(args):
    # Ground truth
    #annFile = '/nightowls/annotations/nightowls_validation.json'
    annFile = args.groundtruth_jsonfile_path

    # Detections
    #resFile = '../sample-Faster-RCNN-nightowls_validation.json'
    resFile = args.detection_jsonfile_path

    output_dir = args.output_dir

    output_file_path = 'results.txt'
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = join(output_dir, output_file_path)

    ## running evaluation
    res_file = open(output_file_path, "w")
    for id_setup in range(0,4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        cocoEval.summarize(id_setup,res_file)

    res_file.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--groundtruth_jsonfile_path', type=str, default=None)
    parser.add_argument('-dt', '--detection_jsonfile_path', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)