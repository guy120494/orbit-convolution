# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random
import pathlib
import tensorflow as tf
from tensorflow import keras

from models.BasicModel import BasicModel
from models.OrbitModel import OrbitModel
from preprocessing.fsdd import get_spectrograms

EPOCHS = 1


def randomly_rotate(x):
    number_of_rotations = random.choice([0, 1, 2, 3])
    return tf.image.rot90(x, k=number_of_rotations)


def grad(model, loss_fn, inputs, targets, training=True):
    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=training)
        loss_value = loss_fn(y_true=targets, y_pred=y_pred)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), y_pred


def train_model(model, train_set, rotate_train=False, epochs=EPOCHS):
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Iterate over the batches of the dataset.
        for x_batch_train, y_true in train_set:
            if rotate_train:
                x_batch_train = randomly_rotate(x_batch_train)
            loss_value, grads, y_pred = grad(model, loss_fn, x_batch_train, y_true)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y_true, y_pred)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        # if (epoch + 1) % 10 == 0:
        #     print("Epoch {}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1,
        #                                                             epoch_loss_avg.result(),
        #                                                             epoch_accuracy.result()))


def test_model(model, test_set, rotate_test=False):
    test_accuracy = tf.keras.metrics.Accuracy()
    for (x, y) in test_set:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        if rotate_test:
            x = randomly_rotate(x)
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    return test_accuracy.result().numpy()


if __name__ == '__main__':
    data_dir = pathlib.Path().absolute() / 'datasets' / 'spectrograms'
    train_set = get_spectrograms(str(data_dir / 'train'))
    test_set = get_spectrograms(str(data_dir / 'test'))

    model = BasicModel()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit(x=train_set, epochs=25, verbose=0)

    result = model.evaluate(test_set, return_dict=True)
    print(result)
    print(model.summary())

    orbit_model = OrbitModel(axis=1)
    orbit_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

    orbit_model.fit(x=train_set, epochs=25, verbose=0)

    result = orbit_model.evaluate(test_set, return_dict=True)
    print(result)
    print(orbit_model.summary())
