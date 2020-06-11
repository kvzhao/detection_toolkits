import os
import sys
import json
import cv2
import copy

from os.path import join

from tqdm import tqdm

def load_json(path):
    with open(path, 'r') as fp:
        jsonfile = json.load(fp)
    return jsonfile

def loadImg(imgpath):
    """
    :param imgpath: the path of image to load
    :return: loaded img object
    """
    print('filename:', imgpath)
    if not os.path.exists(imgpath):
        print('Can not find {}, please check local dataset!'.format(imgpath))
        return None
    img = cv2.imread(imgpath)
    return img

def savesubimage(output_dir, img, subimgname, coordinates):
    left, up, right, down = coordinates
    subimg = copy.deepcopy(img[up: down, left: right])
    outdir = os.path.join(output_dir, subimgname)
    cv2.imwrite(outdir, subimg)

def main(args):

    image_dir = args.image_dir

    split_annotation_path = args.split_annotation_path
    output_dir = args.output_dir
    output_anno_path = os.path.join(output_dir, 'crowd_split.json')

    subimage_width = args.subimage_width
    subimage_height = args.subimage_height
    gap = args.gap

    slide_width = subimage_width - gap
    slide_height = subimage_height - gap

    os.makedirs(output_dir, exist_ok=True)
    output_image_dir = join(output_dir, 'images')
    os.makedirs(output_image_dir, exist_ok=True)

    split_annos = load_json(split_annotation_path)

    scale = args.scale

    # subimage coco image headers
    json_dict = {
        'images': [],
        'annotations': [],
        'categories': [],
    }

    split_image_id = 0
    for image_name, RoIList in split_annos.items():

        #imgwidth = imagedict['image size']['width']
        #imgheight = imagedict['image size']['height']
        #image_id = imagedict['image id']

        image_path = join(image_dir, image_name)
        Img = loadImg(image_path)

        if Img is None:
            print('{} is None, skip'.format(image_path))
            continue

        if scale != 1:
            resizeimg = cv2.resize(Img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = Img

        imgheight, imgwidth = resizeimg.shape[:2]

        for coordinates in RoIList:
            # split image and annotation in sliding window manner
            outbasename = image_name.replace('/', '_').split('.')[0] + '___' + str(scale) + '__'
            subimageannos = {}
            left, up, right, down = coordinates

            subimgname = outbasename + str(left) + '__' + str(up) + '.jpg'

            subimgage_width_ratio = (right - left + 1) / imgwidth
            subimgage_height_ratio = (down - up + 1) / imgheight
            print('subimage: {}, width, height ratio: {}, {}'.format(subimgname,
                subimgage_width_ratio, subimgage_height_ratio))
            # 
            savesubimage(output_image_dir, resizeimg, subimgname, coordinates)
            image_info = {
                'file_name': subimgname,
                'height': down - up + 1,
                'width': right - left + 1,
                'id': split_image_id,
            }
            json_dict['images'].append(image_info)
            split_image_id += 1
            # TODO: Directly save as coco format
            subimageannos[subimgname] = {
                "image size": {
                    "height": down - up + 1,
                    "width": right - left + 1
                },
            }

    json_output_path = join(output_dir, 'coco_anno.json')
    with open(json_output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(json_output_path))

    print('Done.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--image_dir', type=str, default='/home/kv_zhao/datasets/panda/image_test/', help='')
    parser.add_argument('-od', '--output_dir', type=str, default=None, help='')
    parser.add_argument('-sp', '--split_annotation_path', type=str, default=None, help='')
    parser.add_argument('-gt', '--panda_annotation_path', type=str,
        default='/home/kv_zhao/datasets/panda/annoJSONs/person_bbox_test.json', help='')
    parser.add_argument('--scale', type=float, default=1.0, help='Image rescale')
    parser.add_argument('-sw', '--subimage_width', type=int, default=2048)
    parser.add_argument('-sh', '--subimage_height', type=int, default=1024)
    parser.add_argument('--gap', type=int, default=100)
    args = parser.parse_args()
    main(args)
