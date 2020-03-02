

from absl import app
from absl import flags
import logging

import tensorflow as tf
#import tensorflow.compat.v2 as tf
from official.vision.detection.dataloader.tf_example_decoder import TfExampleDecoder


flags.DEFINE_string('record_path', '/tmp/train', 'Path to .record file.')
FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def data_sanity_check(data):
    image = data['image']
    image_height = data['height']
    image_width = data['width']
    gt_bboxes = data['groundtruth_boxes']
    # len of the shape
    print(gt_bboxes)


def main(_):

    print(FLAGS.record_path)
    dataset = tf.data.TFRecordDataset(FLAGS.record_path)
    #dataset = tf.data.Dataset.list_files(FLAGS.record_path)

    decoder = TfExampleDecoder()
    dataset = dataset.map(decoder.decode)
    dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for data in dataset:
        data_sanity_check(data)



if __name__ == '__main__':
    logger = tf.get_logger()
    logger.setLevel(logging.INFO)
    app.run(main)
