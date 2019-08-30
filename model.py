"""This module provides the class implimentation of the network."""
import tensorflow as tf
from tensorflow import keras


class LandmarkModel(keras.Model):

    def __init__(self, output_size):
        super(LandmarkModel, self).__init__(name='landmark_model')

        # The model may take variable number of landmarks.
        self.output_size = output_size

        # The model is composed of multiple layers that best be defined in the
        # init function.

        # Convolutional layers.
        self.conv_1 = keras.layers.Conv2D(filters=32,
                                          kernel_size=(3, 3),
                                          activation='relu')
        self.conv_2 = keras.layers.Conv2D(filters=64,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding='valid',
                                          activation='relu')
        self.conv_3 = keras.layers.Conv2D(filters=64,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=tf.nn.relu)
        self.conv_4 = keras.layers.Conv2D(filters=64,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding='valid',
                                          activation='relu')
        self.conv_5 = keras.layers.Conv2D(filters=64,
                                          kernel_size=[3, 3],
                                          strides=(1, 1),
                                          padding='valid',
                                          activation='relu')
        self.conv_6 = keras.layers.Conv2D(filters=128,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding='valid',
                                          activation='relu')
        self.conv_7 = keras.layers.Conv2D(filters=128,
                                          kernel_size=[3, 3],
                                          strides=(1, 1),
                                          padding='valid',
                                          activation='relu')
        self.conv_8 = keras.layers.Conv2D(filters=256,
                                          kernel_size=[3, 3],
                                          strides=(1, 1),
                                          padding='valid',
                                          activation='relu')

        # Pooling layers.
        self.pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             padding='valid')
        self.pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             padding='valid')
        self.pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                             strides=(2, 2),
                                             padding='valid')
        self.pool_4 = keras.layers.MaxPool2D(pool_size=[2, 2],
                                             strides=(1, 1),
                                             padding='valid')

        # Dense layers.
        self.dense_1 = keras.layers.Dense(units=1024,
                                          activation='relu',
                                          use_bias=True)
        self.dense_2 = keras.layers.Dense(units=self.output_size,
                                          activation=None,
                                          use_bias=True)

        # Flatten layers.
        self.flatten_1 = keras.layers.Flatten()

    def call(self, inputs):
        """Network forward definition. Using the layers defined in the `__init__`
        Args:
            inputs: input of the network.
        """

        # |== Layer 1 ==|
        inputs = self.conv_1(inputs)
        inputs = self.pool_1(inputs)

        # |== Layer 2 ==|
        inputs = self.conv_2(inputs)
        inputs = self.conv_3(inputs)
        inputs = self.pool_2(inputs)

        # |== Layer 3 ==|
        inputs = self.conv_4(inputs)
        inputs = self.conv_5(inputs)
        inputs = self.pool_3(inputs)

        # |== Layer 4 ==|
        inputs = self.conv_6(inputs)
        inputs = self.conv_7(inputs)
        inputs = self.pool_4(inputs)

        # |== Layer 5 ==|
        inputs = self.conv_8(inputs)

        # |== Layer 6 ==|
        inputs = self.flatten_1(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.dense_2(inputs)

        return inputs

    def compute_output_shape(self, input_shape):
        # Override this function to use the subclassed model as part of a
        # functional-style model.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_size
        return tf.TensorShape(shape)
