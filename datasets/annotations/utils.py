
import os
import sys
import cv2
import random
from bounding_box import bounding_box as bb

from os.path import join, basename

def get_files(path, ext=('.png', '.jpg')):
  # ext is str or tuple
  files = []
  for (dir_path, _, file_names) in os.walk(path):
    for file_name in file_names:
      if file_name.endswith(ext):
        files.append(join(dir_path, file_name))
  return files


def read_image(filepath):
  img = cv2.imread(filepath)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def save_image(img, filepath):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.imwrite(filepath, img)


def draw_bbox(img, bbox, conf=None, color='red'):
  boxtext = str(conf) if conf is not None else 'face'
  bb.add(img, *bbox, boxtext, color)
  return img


def xywh2xyxy(bbox):
  return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]


def xyxy2xywh(bbox):
  return [bbox[0], bbox[1], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1])]


def crop_image(img, region):
  # region in x,y,w,h
  r = [int(v) for v in xywh2xyxy(region)]
  return img[r[1]:r[3], r[0]:r[2]]


# TODO:
def plot_image_grid(file_names, output_path, lx=10, ly=10):
  plt.tight_layout()
  f, axarr = plt.subplots(lx, ly, figsize=(18, 18))
  f.set_constrained_layout(False)
  for i, name in enumerate(random.sample(file_names, lx*ly)):
      s = imginfo[imginfo['filename'] == name]
      img_path  = s['img_path'].values[0]
      img = read_image(img_path)
      gx = i % lx
      gy = i // ly
      axarr[gx, gy].grid(False)
      axarr[gx, gy].get_xaxis().set_visible(False)
      axarr[gx, gy].get_yaxis().set_visible(False)
      axarr[gx, gy].imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.axis('off')
  #plt.imshow(img)
  plt.savefig(output_path, bbox_inches='tight')