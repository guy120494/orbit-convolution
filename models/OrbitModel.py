from enum import Enum, auto

import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling1D, Dropout
from tensorflow.python.keras.models import Model

from layers.OrbitConvolution import OrbitSumConvolution, OrbitMeanConvolution, OrbitMaxConvolution


class InvarianceType(Enum):
    MAX = auto()
    MEAN = auto()
    SUM = auto()


class OrbitModel(Model):

    def __init__(self, axis: int = 0, invariance_type: InvarianceType = InvarianceType.SUM):
        super(OrbitModel, self).__init__()
        self.invariance_type = invariance_type
        self.axis = axis
        self.first_cnn_layer = None
        self.second_cnn_layer = self.get_orbit_layer()
        self.max_pooling = MaxPooling1D(pool_size=2)
        self.first_dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.first_dense = Dense(930, activation='relu')
        self.second_dropout = Dropout(0.5)
        self.second_dense = Dense(10, activation='softmax')

    def get_orbit_layer(self):
        num_filters = 256
        if self.invariance_type == InvarianceType.MAX:
            return OrbitMaxConvolution(name="orbit", kernel_size=3, num_filters=num_filters, axis=self.axis)
        elif self.invariance_type == InvarianceType.MEAN:
            return OrbitMeanConvolution(name="orbit", kernel_size=3, num_filters=num_filters, axis=self.axis)
        elif self.invariance_type == InvarianceType.SUM:
            return OrbitSumConvolution(name="orbit", kernel_size=3, num_filters=num_filters, axis=self.axis)

    def build(self, input_shape):
        self.first_cnn_layer = Conv2D(name="convolution", filters=32, kernel_size=(3, 3), activation='relu',
                                      input_shape=input_shape)
        super(OrbitModel, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.first_cnn_layer(inputs)
        x = self.second_cnn_layer(x)
        x = tf.nn.relu(x)
        x = self.max_pooling(x)
        x = self.first_dropout(x, training=training)
        x = self.flatten(x)
        x = self.first_dense(x)
        x = self.second_dropout(x, training=training)
        return self.second_dense(x)

    def get_config(self):
        pass
