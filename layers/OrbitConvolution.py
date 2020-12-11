import tensorflow as tf
from tensorflow.keras.layers import Layer


class OrbitSumConvolution(Layer):
    def __init__(self, kernel_size: int, num_filters: int, axis: int = 0, **kwargs):
        self.axis = axis
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.kernel = None
        super(OrbitSumConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape([self.kernel_size, input_shape[-1], self.num_filters])

        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='he_uniform')
        super(OrbitSumConvolution, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.math.reduce_sum(x, axis=self.axis + 1)
        return tf.nn.conv1d(x, self.kernel, padding='SAME', stride=1)

    def compute_output_shape(self, input_shape):
        return self.kernel_size, input_shape[-(self.axis + 1)], self.num_filters


class OrbitMaxConvolution(Layer):
    def __init__(self, kernel_size: int, num_filters: int, axis: int = 0, **kwargs):
        self.axis = axis
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.kernel = None
        super(OrbitMaxConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape([self.kernel_size, input_shape[-1], self.num_filters])

        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='he_uniform')
        super(OrbitMaxConvolution, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.math.reduce_max(x, axis=self.axis + 1)
        return tf.nn.conv1d(x, self.kernel, padding='SAME', stride=1)

    def compute_output_shape(self, input_shape):
        return self.kernel_size, input_shape[-(self.axis + 1)], self.num_filters


class OrbitMeanConvolution(Layer):
    def __init__(self, kernel_size: int, num_filters: int, axis: int = 0, **kwargs):
        self.axis = axis
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.kernel = None
        super(OrbitMeanConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape([self.kernel_size, input_shape[-1], self.num_filters])

        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='he_uniform')
        super(OrbitMeanConvolution, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.math.reduce_mean(x, axis=self.axis + 1)
        return tf.nn.conv1d(x, self.kernel, padding='SAME', stride=1)

    def compute_output_shape(self, input_shape):
        return self.kernel_size, input_shape[-(self.axis + 1)], self.num_filters