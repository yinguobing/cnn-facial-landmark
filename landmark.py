"""
Convolutional Neural Network for facial landmarks detection.
"""
import argparse

import cv2
import numpy as np
import tensorflow as tf

from model import LandmarkModel

# Add arguments parser to accept user specified arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--train_record', default='train.record', type=str,
                    help='Training record file')
parser.add_argument('--val_record', default='validation.record', type=str,
                    help='validation record file')
parser.add_argument('--model_dir', default='train', type=str,
                    help='training model directory')
parser.add_argument('--export_dir', default=None, type=str,
                    help='directory to export the saved model')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='training steps')
parser.add_argument('--num_epochs', default=None, type=int,
                    help='epochs of training dataset')
parser.add_argument('--batch_size', default=16, type=int,
                    help='training batch size')
parser.add_argument('--raw_input', default=False, type=bool,
                    help='Use raw tensor as model input.')


# CAUTION: The image width, height and channels should be consist with your
# training data. Here they are set as 128 to be complied with the tutorial.
# Mismatching of the image size will cause error of mismatching tensor shapes.
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3


def cnn_model_fn(features, labels, mode):
    """
    The model function for the network.
    """
    # Construct the network.
    model = LandmarkModel(output_size=68*2)
    logits = model(features)

    # Make prediction for PREDICATION mode.
    predictions = logits
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss using mean squared error.
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    # Create a tensor logging purposes.
    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    # Configure the train OP for TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
    else:
        train_op = None

    # Create a metric.
    rmse_metrics = tf.metrics.root_mean_squared_error(
        labels=labels,
        predictions=predictions)
    metrics = {'eval_mse': rmse_metrics}

    # A tensor for metric logging
    tf.identity(rmse_metrics[1], name='root_mean_squared_error')
    tf.summary.scalar('root_mean_squared_error', rmse_metrics[1])

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )


def _parse_function(record):
    """
    Extract data from a `tf.Example` protocol buffer.
    """
    # Defaults are not specified since both keys are required.
    keys_to_features = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'label/marks': tf.FixedLenFeature([136], tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)

    # Extract features from single example
    image_decoded = tf.image.decode_image(parsed_features['image/encoded'])
    image_reshaped = tf.reshape(
        image_decoded, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    points = tf.cast(parsed_features['label/marks'], tf.float32)

    return image_reshaped, points


def input_fn(record_file, batch_size, num_epochs=None, shuffle=True):
    """
    Input function required for TensorFlow Estimator.
    """
    dataset = tf.data.TFRecordDataset(record_file)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    # Make dataset iterator.
    iterator = dataset.make_one_shot_iterator()

    # Return the feature and label.
    image, label = iterator.get_next()
    return image, label


def serving_input_receiver_fn():
    """An input function for TensorFlow Serving."""

    def _preprocess_image(image_bytes):
        """Preprocess a single raw image."""
        image = tf.image.decode_jpeg(image_bytes, channels=IMG_CHANNEL)
        image.set_shape((None, None, IMG_CHANNEL))
        image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH],
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=False)
        return image
    image_bytes_list = tf.compat.v1.placeholder(
        shape=[None], dtype=tf.string,
        name='encoded_image_string_tensor')
    image = tf.map_fn(_preprocess_image, image_bytes_list,
                      dtype=tf.float32, back_prop=False)

    return tf.estimator.export.TensorServingInputReceiver(
        features=image,
        receiver_tensors={'image_bytes': image_bytes_list})


def serving_input_tensor_receiver_fn():
    """An input function accept raw tensors."""
    def _preprocess_image(image_tensor):
        """Preprocess a single raw image tensor."""
        image = tf.image.resize_images(image_tensor, [IMG_HEIGHT, IMG_WIDTH],
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=False)
        return image

    image_tensor = tf.compat.v1.placeholder(
        shape=[None, None, None, 3], dtype=tf.uint8,
        name='image_tensor')
    image = tf.map_fn(_preprocess_image, image_tensor,
                      dtype=tf.float32, back_prop=False)

    return tf.estimator.export.TensorServingInputReceiver(
        features=image,
        receiver_tensors={'image': image_tensor})


def main(unused_argv):
    """Train, eval and export the model."""
    # Parse the arguments.
    args = parser.parse_args(unused_argv[1:])

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=args.model_dir)

    # Train for N steps.
    tf.logging.info('Starting to train.')
    estimator.train(
        input_fn=lambda: input_fn(record_file=args.train_record,
                                  batch_size=args.batch_size,
                                  num_epochs=args.num_epochs,
                                  shuffle=True),
        steps=args.train_steps)

    # Do evaluation after training.
    tf.logging.info('Starting to evaluate.')
    evaluation = estimator.evaluate(
        input_fn=lambda: input_fn(record_file=args.val_record,
                                  batch_size=1,
                                  num_epochs=1,
                                  shuffle=False))
    print(evaluation)

    # Export trained model as SavedModel.
    receiver_fn = serving_input_tensor_receiver_fn if args.raw_input else serving_input_receiver_fn
    if args.export_dir is not None:
        estimator.export_savedmodel(args.export_dir, receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
