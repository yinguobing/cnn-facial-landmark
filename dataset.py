"""This module provides the functions to build a TensorFlow dataset."""

import tensorflow as tf


def _parse(example):
    """Extract data from a `tf.Example` protocol buffer.
    Args:
        example: a protobuf example.

    Returns:
        a parsed data and label pair.
    """
    # Defaults are not specified since both keys are required.
    keys_to_features = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'label/marks': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example, keys_to_features)

    # Extract features from single example
    image_decoded = tf.image.decode_image(parsed_features['image/encoded'])
    image_float = tf.cast(image_decoded, tf.float32)

    points = tf.io.parse_tensor(parsed_features['label/marks'], tf.float64)
    points = tf.reshape(points, [-1])
    points = tf.cast(points, tf.float32)

    return image_float, points


def get_parsed_dataset(record_file, batch_size, shuffle=True):
    """Return a parsed dataset for model.
    Args:
        record_file: the TFRecord file.
        batch_size: batch size.
        shuffle: whether to shuffle the data.

    Returns:
        a parsed dataset.
    """
    # Init the dataset from the TFRecord file.
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.map(_parse, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
