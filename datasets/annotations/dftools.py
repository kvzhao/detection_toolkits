
import pandas as pd
import numpy as np
import random
import json

from pycocotools.coco import COCO

#from nptools.np_box_list_ops import iou
#from nptools.np_box_list import BoxList
from utils import xywh2xyxy
from utils import xyxy2xywh

class DataframeFields:
  file_name = 'file_name'
  img_path = 'img_path'
  width = 'width'
  height = 'height'
  bboxes = 'bboxes'
  bbox = 'bbox'
  local_bbox = 'local_bbox'
  bbox_overlap = 'bbox_overlap'
  loose_region = 'loose_region'
  instance_name = 'instance_name'
  img_size = 'img_size'
  img_aspect_ratio = 'img_aspect_ratio'
  landmarks = 'landmarks'
  local_landmarks = 'local_landmarks'

Fields = DataframeFields


"""
  The given dataframe must have 5 columns:
    'file_name', 'img_path', 'width', 'height', 'bboxes'
"""

def compute_statistics(df):
  """Statistic measures
    - image_size
    - face_size
    - area ratio
    - image aspect_ratio
    - face aspect_ratio
    - total amount
  """
  print(df.describe())


def compute_img_size(df):
  df[Fields.img_size] = df.apply(lambda x: np.sqrt(x.height * x.width), axis=1)
  return df


def compute_img_aspect_ratio(df):
  df[Fields.img_aspect_ratio] = df.apply(
    lambda x: np.maximum(x.height/x.width, x.width/x.height),
    axis=1)
  return df


def compute_bbox_overlap(df):
  df[Fields.bbox_overlap] = df.apply(
    lambda x: calc_bboxes_overlap_per_image(x.bboxes), axis=1)
  return df


def sample_file_names(df, n=1):
  names = random.sample(list(df[df.num_face ==2][Fields.file_name].values), n)
  if n == 1:
    names = names[0]
  return names


def get_bboxes_by_name(df, name):
  if isinstance(name, str):
    return df[df.file_name == name].bboxes.values[0]


def filter_overlapped_image(df, thres=0.01):
  filtered = df[df.bbox_overlap <= thres]
  print('[overlap] #of images are filter out: {}'.format(len(df) - len(filtered)))
  return filtered


def filter_image_size(df, min_img_size=32, is_crop=True):
  #if Fields.img_size not in df:
  # must re-compute
  df = compute_img_size(df, is_crop)
  filtered = df[df.img_size >= min_img_size]
  print('[img_size] #of images are filter out: {}'.format(len(df) - len(filtered)))
  return filtered


def filter_image_aspect_ration(df, max_ratio=3, is_crop=True):
  #if Fields.img_aspect_ratio not in df:
  df = compute_img_aspect_ratio(df, is_crop)
  filtered = df[df.img_aspect_ratio <= max_ratio]
  print('[img_aspect_ratio] #of images are filter out: {}'.format(len(df) - len(filtered)))
  return filtered


def list_rounding(df, columns=[Fields.bbox,
                               Fields.bboxes,
                               Fields.local_bbox,
                               Fields.loose_region]):
  for c in columns:
    if c not in df:
      continue
    df[c] = df.apply(lambda x: [round(v) for v in x[c]], axis=1)
  return df


def save(df, path):
  # in HDF5 format
  with pd.HDFStore(path, 'w') as store:
    store['df'] = df
  return True


def load(path):
  # in HDF5 format
  with pd.HDFStore(path, 'r') as store:
    df = store['df']
  return df


def split2(df, ratio=0.8):
  """split dataframe into 2 indep dfs
    Args:
      df: Original
      ratio: a float
    Returns
      df1, df2
  """
  df1 = df.sample(frac=ratio, random_state=1234)
  df2 = df.drop(df1.index)
  return df1, df2


def merge(df1, df2):
  # merge two dataframes to dump the same coco annotation
  for c in [Fields.file_name, Fields.img_path, Fields.height, Fields.width, Fields.bboxes]:
    if c not in df1:
      print('WARNING: {} is missing in df1')
    if c not in df2:
      print('WARNING: {} is missing in df2')
  df = pd.concat([df1, df2])
  print('Merged df has {} rows from df1 ({}) + df2 ({})'.format(len(df), len(df1), len(df2)))
  return df


def dump_coco(df, path):
  # dump dataframe as coco annotation json
  # This function is really slow
  json_dict = {
      'images': [],
      'annotations': [],
      'categories': [
        {
          'supercategory': 'face',
          'id': 1,
          'name': 'face',
        }
      ],
  }
  image_id = 1
  bbox_id = 1
  ignore = 0 
  category_id = 1
  image = {}

  has_landmark = Fields.landmarks in df.columns
  print('has_landmark:', has_landmark)

  def _convert_landmark_format(landmark):
    # [x,y,x,y ...] -> [x,y,v,x,y,v ...]
    pass

  print('Start processing {}...'.format(path))
  for name in list(df.file_name.values):
    sample = df[df.file_name == name]
    image_info = {
      'file_name': name,
      'height': int(sample.height.values[0]),
      'width': int(sample.width.values[0]),
      'id': image_id,
    }
    json_dict['images'].append(image_info)
    for bboxes in list(sample.bboxes.values):
      # TODO: check bounding boxes
      if not isinstance(bboxes[0], list):
        bboxes = [bboxes]
      # TODO: Add landmarks if exists, and add visibility
      for bbox in bboxes:
        xmin, ymin, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        int_box = [xmin, ymin, w, h]
        annotation = {
          'area': w * h,
          'iscrowd': ignore,
          'image_id': image_id,
          'bbox': int_box,
          'category_id': category_id,
          'id': bbox_id,
          'ignore': ignore,
          'segmentation': [],
        }
        json_dict['annotations'].append(annotation)
        bbox_id += 1
    image_id += 1
  with open(path, 'w') as json_fp:
      json_str = json.dumps(json_dict)
      json_fp.write(json_str)
  print('Done, save to {}'.format(path))


def from_coco(gt_path, dt_path=None):
  """Get dataframe from coco json fiie
      Two json files are needed when loading predicted results.
    Args:
      gt_path: string
      dt_path: string
    Return:
      df
  """
  def _dt2df(Gt, Dt):
      gt_bboxes = []
      file_names = []
      widths = []
      heights = []
      dt_bboxes = []
      dt_scores = []

      for img_id in Gt.getImgIds():
          infos = Gt.loadImgs(img_id)
          annos = Gt.loadAnns(Gt.getAnnIds(img_id))
          preds = Dt.loadAnns(Dt.getAnnIds(img_id))
          file_names.append(infos[0]['file_name'])
          widths.append(infos[0]['width'])
          heights.append(infos[0]['height'])
          gt_bboxes.append(list(map(lambda x: x['bbox'], annos)))
          dt_bboxes.append(list(map(lambda x: x['bbox'], preds)))
          dt_scores.append(list(map(lambda x: x['score'], preds)))
      return pd.DataFrame(
          {
              'file_name': file_names,
              'height': heights,
              'width': widths,
              'bboxes': gt_bboxes,
              'dt_bboxes': dt_bboxes,
              'dt_scores': dt_scores,
          }
      )

  def _gt2df(Gt):
      gt_bboxes = []
      file_names = []
      widths = []
      heights = []

      for img_id in Gt.getImgIds():
          infos = Gt.loadImgs(img_id)
          annos = Gt.loadAnns(Gt.getAnnIds(img_id))
          file_names.append(infos[0]['file_name'])
          widths.append(infos[0]['width'])
          heights.append(infos[0]['height'])
          gt_bboxes.append(list(map(lambda x: x['bbox'], annos)))
      return pd.DataFrame(
          {
              'file_name': file_names,
              'height': heights,
              'width': widths,
              'bboxes': gt_bboxes,
          }
      )

  if dt_path is not None:
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    return _dt2df(cocoGt, cocoDt)
  else:
    cocoGt = COCO(gt_path)
    return _gt2df(cocoGt)
