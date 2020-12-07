import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer


class OrbitConvolution(Layer):
    def __init__(self, kernel_size: int, num_filters: int, axis: int = 0, **kwargs):
        self.axis = axis
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.kernel = None
        super(OrbitConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape([self.kernel_size, input_shape[-(self.axis + 2)], self.num_filters])
        # only have a 3x3 kernel
        # shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='glorot_uniform')
        super(OrbitConvolution, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.math.reduce_sum(x, axis=self.axis + 2)
        return keras.backend.conv1d(x, self.kernel, padding='same')

    def compute_output_shape(self, input_shape):
        return self.kernel_size, input_shape[-(self.axis + 2)], self.num_filters
