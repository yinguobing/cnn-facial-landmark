"""Simple image facial landmark detection.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs 68 facial
landmark points.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform landmark detection.

https://yinguobing.com/
"""

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image

from tensorflow.contrib.layers.python import ops

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://yinguobing.com/content/'
# pylint: enable=line-too-long


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
        FLAGS.model_dir, 'frozen_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_file):
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image_file):
        tf.logging.fatal('File does not exist %s', image_file)
    # image_data = tf.gfile.FastGFile(image, 'rb').read()
    # image = tf.image.decode_image(image_data)
    image_array = Image.open(image_file)
    image_array = np.array(image_array)

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        logits_tensor = sess.graph.get_tensor_by_name('logits/BiasAdd:0')
        predictions = sess.run(logits_tensor,
                               {'input_image_tensor:0': image_array})
        marks = np.array(predictions).flatten() * 128
        marks = np.reshape(marks, (-1, 2))

        import cv2
        img = cv2.imread(image_file)
        for mark in marks:
            cv2.circle(img, (int(mark[0]), int(
                mark[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
        img = cv2.resize(img, (512, 512))
        cv2.imshow('preview', img)
        cv2.waitKey()


# def maybe_download_and_extract():
#     """Download and extract model tar file."""
#     dest_directory = FLAGS.model_dir
#     if not os.path.exists(dest_directory):
#         os.makedirs(dest_directory)
#     filename = DATA_URL.split('/')[-1]
#     filepath = os.path.join(dest_directory, filename)
#     if not os.path.exists(filepath):
#         def _progress(count, block_size, total_size):
#             sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#                 filename, float(count * block_size) / float(total_size) * 100.0))
#             sys.stdout.flush()
#         filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#         print()
#         statinfo = os.stat(filepath)
#         print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
#     tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    # maybe_download_and_extract()
    image = (FLAGS.image_file if FLAGS.image_file else
             os.path.join(FLAGS.model_dir, 'face.jpg'))
    run_inference_on_image(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # graph.pbtxt:
    #     Binary representation of the GraphDef protocol buffer.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/robin/Desktop/cnn-facial-landmark/saved_model',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
