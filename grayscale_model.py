import inputs
import model_tools as mt
import tensorflow as tf
import numpy as np
import keras
from keras import layers  # noqa

IMG_SIZE = 150


def get_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu,
                            input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(34, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', "sparse_categorical_crossentropy"])

    return model


def run(model, plot=False, test=False, save=False):
    grays, labels = inputs.load("GraysCompressedData")

    indices = np.arange(grays.shape[0])
    np.random.seed(0)
    np.random.shuffle(indices)

    grays = grays[indices]
    labels = labels[indices]
    grays = grays.astype("float16")
    grays = grays[..., np.newaxis]
    grays = grays/255.0

    train_data = grays[:22000]
    train_labels = labels[:22000]
    test_data = grays[22000:]
    test_labels = labels[22000:]

    print("Train Data:", train_data.shape)
    print("Train Label:", train_labels.shape)
    print("Test Data:", test_data.shape)
    print("Train Label:", test_labels.shape)

    trained_model, history = mt.train_model(
        model, train_data, train_labels, test_data, test_labels, epoch=7)

    if (save):
        mt.save_model(trained_model, "grayscale_model")
    if (test):
        mt.test_model(trained_model, test_data, test_labels)
    if (plot):
        mt.plot_history(history)
