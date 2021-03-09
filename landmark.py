"""
Convolutional Neural Network for facial landmarks detection.
"""
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import get_parsed_dataset
from model import build_landmark_model

# Introduce arguments parser to give user the flexibility to tune the process.
parser = argparse.ArgumentParser()
parser.add_argument('--train_record', default='train.record', type=str,
                    help='Training record file')
parser.add_argument('--val_record', default='validation.record', type=str,
                    help='validation record file')
parser.add_argument('--epochs', default=1, type=int,
                    help='epochs for training')
parser.add_argument('--batch_size', default=16, type=int,
                    help='training batch size')
parser.add_argument('--export_only', default=False, type=bool,
                    help='Save the model without training and evaluation.')
parser.add_argument('--eval_only', default=False, type=bool,
                    help='Do evaluation without training.')
args = parser.parse_args()


if __name__ == '__main__':
    # Checkpoint is used to track the training model so it could be restored
    # later to resume training.
    checkpoint_dir = "checkpoints"

    # Besides checkpoint, `saved_model` is another way to save the model for
    # inference or optimization.
    export_dir = "exported"

    # The log directory for tensorboard.
    log_dir = "logs"

    # The input image's width, height and channels should be consist with your
    # training data. Here they are set to be complied with the tutorial.
    input_shape = (128, 128, 3)

    # The number of facial landmarks the model should output. By default the
    # marks are in 2D space.
    num_marks = 68

    # Create the Model
    model = build_landmark_model(input_shape=input_shape,
                                 output_size=num_marks*2)

    # Prepare for training. First restore the model if any checkpoint file available.
    if not tf.io.gfile.exists(checkpoint_dir):
        tf.io.gfile.mkdir(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
    else:
        print("Checkpoint not found. Model weights will be initialized randomly.")

    # Sometimes the user only want to save the model. Skip training in this case.
    if args.export_only:
        if not tf.io.gfile.exists(export_dir):
            tf.io.gfile.mkdir(export_dir)

        if latest_checkpoint is None:
            print("Warning: Model not restored from any checkpoint.")

        print("Saving model to {} ...".format(export_dir))
        model.save(export_dir, include_optimizer=False)
        print("Model saved at: {}".format(export_dir))
        quit()

    # Finally, it's time to train the model.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.mean_squared_error)

    # Construct a dataset for evaluation.
    dataset_val = get_parsed_dataset(record_file=args.val_record,
                                     batch_size=args.batch_size,
                                     shuffle=False)

    # If evaluation is required only.
    if args.eval_only:
        print('Starting to evaluate.')
        evaluation = model.evaluate(dataset_val)
        quit()

    # To save and log the training process, we need some callbacks.
    callback_tb = keras.callbacks.TensorBoard(log_dir=log_dir)
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+"/landmark",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
    callbacks = [callback_tb, callback_checkpoint]

    # Get the training data ready.
    dataset_train = get_parsed_dataset(record_file=args.train_record,
                                       batch_size=args.batch_size,
                                       shuffle=True)

    model.fit(dataset_train, validation_data=dataset_val, epochs=args.epochs,
              callbacks=callbacks)
