
import os
import sys
import json

from os.path import join
from os.path import basename

from pycocotools.coco import COCO



FOLDER_NAME_TABLE = {
  '14': '14_OCT_Habour',
  '15': '15_Nanshani_Park',
  '16': '16_Primary_School',
  '17': '17_New_Zhongguan',
  '18': '18_Xili_Street',
}


def xyxy2xywh(bbox):
  return [bbox[0], bbox[1], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1])]

def get_files(path, ext=('.json')):
  # ext is str or tuple
  files = []
  for (dir_path, _, file_names) in os.walk(path):
    for file_name in file_names:
      if file_name.endswith(ext):
        files.append(join(dir_path, file_name))
  return files


def main(args):

  data_dir = args.data_dir
  category_id = args.category_id
  reference_image_info = args.reference_image_info
  output_path = args.output_dir

  reference_image_info = json.load(open(reference_image_info, 'r'))

  json_file_paths = get_files(data_dir, ('.json'))
  # json_file_names = [basename(f) for f in json_file_paths]
  print('Load {} files from {}'.format(len(json_file_paths), data_dir))

  print(json_file_paths)

  json_dict = {
      'images': [],
      'annotations': [],
      'categories': [
        {
          'supercategory': 'person',
          'id': category_id,
          'name': 'person',
        }
      ],
  }
  image_id = 0
  bbox_id = 0
  ignore = 0 
  image = {}

  for json_file_path in json_file_paths:
    with open(json_file_path, 'r') as fp:
      anno = json.load(fp)

    json_file_name = os.path.splitext(basename(json_file_path))[0]
    print(json_file_name)

    scene_id = json_file_name.split('_')[1]
    print(scene_id)

    scene_prefix = FOLDER_NAME_TABLE[scene_id]

    print(scene_prefix)


    if 'imageHeight' not in anno or 'imageWidth' not in anno:
      print(json_file_path)
      continue
    img_height = anno['imageHeight']
    img_width = anno['imageWidth']
    file_name = anno['imagePath']

    #image_file_name = join(scene_prefix, file_name)
    image_file_name = '/'.join([scene_prefix, file_name])
    image_file_id = reference_image_info[image_file_name]['image id']

    image_info = {
      'file_name': image_file_name,
      'height': img_height,
      'width': img_width,
      'id': int(image_file_id),
    }
    json_dict['images'].append(image_info)

    for shape in anno['shapes']:
      try:
        p0, p1 = shape['points']
      except:
        print(shape)
        continue
      bbox = xyxy2xywh([p0[0], p0[1], p1[0], p1[1]])
      _, _, w, h = bbox
      annotation = {
        'area': w * h,
        'iscrowd': ignore,
        'image_id': image_file_id,
        'bbox': bbox,
        'category_id': category_id,
        'id': bbox_id,
        'ignore': ignore,
        'segmentation': [],
      }
      json_dict['annotations'].append(annotation)
      bbox_id += 1
    image_id += 1

  with open(output_path, 'w') as json_fp:
      json_str = json.dumps(json_dict)
      json_fp.write(json_str)
  print('Done, save to {}'.format(output_path))






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-r', '--reference_image_info', type=str, default=None)
    parser.add_argument('-cid', '--category_id', type=int, default=2)
    args = parser.parse_args()
    main(args)
