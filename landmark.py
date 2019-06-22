"""
Convolutional Neural Network for facial landmarks detection.
"""
import cv2
import numpy as np
import tensorflow as tf

# CAUTION
# The image width, height and channels should be consist with your training
# data. Here they are set as 128 to be complied with the tutorial. Mismatching
# of the image size will cause error of mismatching tensor shapes.
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3

# The input functions for train, validation and serving shares a same feature
# for our model function. Name it here.
INPUT_FEATURE = 'image'


def cnn_model_fn(features, labels, mode):
    """
    The model function for the network.
    """
    # |== Layer 0: input layer ==|
    # Input feature x should be of shape (batch_size, image_width, image_height,
    # color_channels). As we will directly using the decoded image tensor of
    # data type int8, a convertion should be performed.
    inputs = tf.cast(features[INPUT_FEATURE], tf.float32)

    # |== Layer 1 ==|

    # Convolutional layer.
    # Computes 32 features using a 3x3 filter with ReLU activation.
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # Pooling layer.
    # First max pooling layer with a 2x2 filter and stride of 2.
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')

    # |== Layer 2 ==|

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # Pooling layer
    # Second max pooling layer with a 2x2 filter and stride of 2.
    pool2 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')

    # |== Layer 3 ==|

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv4 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # Convolutional layer
    # Computes 64 features using a 3x3 filter with ReLU activation.
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # Pooling layer
    # Third max pooling layer with a 2x2 filter and stride of 2.
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')

    # |== Layer 4 ==|

    # Convolutional layer
    # Computes 128 features using a 3x3 filter with ReLU activation.
    conv6 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # Convolutional layer
    # Computes 128 features using a 3x3 filter with ReLU activation.
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # Pooling layer
    # Fourth max pooling layer with a 2x2 filter and stride of 2.
    pool4 = tf.layers.max_pooling2d(
        inputs=conv7,
        pool_size=[2, 2],
        strides=(1, 1),
        padding='valid')

    # |== Layer 5 ==|

    # Convolutional layer
    conv8 = tf.layers.conv2d(
        inputs=pool4,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu)

    # |== Layer 6 ==|

    # Flatten tensor into a batch of vectors
    flatten = tf.layers.flatten(inputs=conv8)

    # Dense layer 1, a fully connected layer.
    dense1 = tf.layers.dense(
        inputs=flatten,
        units=1024,
        activation=tf.nn.relu,
        use_bias=True)

    # Dense layer 2, also known as the output layer.
    logits = tf.layers.dense(
        inputs=dense1,
        units=136,
        activation=None,
        use_bias=True,
        name="logits")

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

    mse_metrics = tf.metrics.root_mean_squared_error(
        labels=labels,
        predictions=predictions)

    metrics = {'eval_mse': mse_metrics}
    # # Create a tensor named train_MSE for logging purposes
    tf.identity(mse_metrics[1], name='eval_mse')
    tf.summary.scalar('eval_mse', mse_metrics[1])

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
    if batch_size != 1:
        dataset = dataset.batch(batch_size)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    # Make dataset iterator.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    image, label = iterator.get_next()
    return {INPUT_FEATURE: image}, label


def _train_input_fn():
    """Function for training."""
    return input_fn(
        record_file="./train.record",
        batch_size=32,
        num_epochs=50,
        shuffle=True)


def _eval_input_fn():
    """Function for evaluating."""
    return input_fn(
        record_file="./validation.record",
        batch_size=2,
        num_epochs=1,
        shuffle=False)


def serving_input_receiver_fn():
    """An input function for TensorFlow Serving."""
    reciever_tensors = tf.placeholder(
        dtype=tf.uint8,
        shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL],
        name='input_image_tensor')
    features = {INPUT_FEATURE: reciever_tensors}

    return tf.estimator.export.ServingInputReceiver(
        receiver_tensors=reciever_tensors,
        features=features)


def main(unused_argv):
    """Train, eval and export the model."""
    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./train")

    # Train for N steps.
    tf.logging.info('Starting to train.')
    estimator.train(input_fn=_train_input_fn, steps=500)

    # Do evaluation after training.
    tf.logging.info('Starting to evaluate.')
    evaluation = estimator.evaluate(input_fn=_eval_input_fn)
    print(evaluation)

    # Export trained model as SavedModel.
    estimator.export_savedmodel('./saved_model', serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
