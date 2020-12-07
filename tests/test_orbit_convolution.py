import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datasets.data_sets_fetcher import get_mnist_data
from layers.OrbitConvolution import OrbitConvolution


class OrbitConvolutionTest(tf.test.TestCase):

    def test_equivariance_in_x_axis(self):
        layer = OrbitConvolution(kernel_size=3, num_filters=1)
        x_train, _, _, _ = get_mnist_data()
        sample = x_train[0]
        sample = np.reshape(sample, [1, *sample.shape])
        sample = tf.convert_to_tensor(sample)

        sample1 = np.roll(sample, (2, 0), axis=(1, 2))
        sample2 = np.roll(sample, (-2, 0), axis=(1, 2))

        result1 = layer(tf.convert_to_tensor(sample1))
        result2 = layer(tf.convert_to_tensor(sample2))

        self.assertAllClose(tf.roll(result1, (-4, 0), axis=(1, 2)), result2, message="Not equivariant")

    def test_invariance_in_x_axis(self):
        layer = OrbitConvolution(kernel_size=3, num_filters=1)
        x_train, _, _, _ = get_mnist_data()
        sample = x_train[0]
        sample = np.reshape(sample, [1, *sample.shape])
        sample = tf.convert_to_tensor(sample)

        sample1 = np.roll(sample, (0, 2), axis=(1, 2))
        sample2 = np.roll(sample, (0, -2), axis=(1, 2))


        result1 = layer(tf.convert_to_tensor(sample1))
        result2 = layer(tf.convert_to_tensor(sample2))

        self.assertAllClose(result1, result2, message="Not invariant")
