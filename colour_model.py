import inputs
import model_tools as mt
import tensorflow as tf
import numpy as np
import keras
from keras import layers  # noqa


IMG_SIZE = 150


def get_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(
            IMG_SIZE, IMG_SIZE, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation=tf.nn.relu),  # consider changing
        keras.layers.Dropout(0.3),
        layers.Dense(34, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
    return model


def run(model, mix=False, plot=False, test=False, save=False):
    colours, labels = inputs.load("BaseCompressedData")

    if (mix):
        augments, aug_labels = inputs.load("AugmentedCompressedData")
        mixed = np.concatenate((colours, augments), axis=0)
        labels = np.concatenate((labels, aug_labels), axis=0)

        indices = np.arange(mixed.shape[0])
        np.random.seed(0)
        np.random.shuffle(indices)

        mixed = mixed[indices]
        labels = labels[indices]
        mixed = mixed.astype("float16")
        mixed = mixed/255.0

        train_data = mixed[:45000]
        train_labels = labels[:45000]
        test_data = mixed[45000:]
        test_labels = labels[45000:]

    else:
        indices = np.arange(colours.shape[0])
        np.random.seed(0)
        np.random.shuffle(indices)

        colours = colours[indices]
        labels = labels[indices]
        colours = colours.astype("float16")
        colours = colours/255.0

        train_data = colours[:22000]
        train_labels = labels[:22000]
        test_data = colours[22000:]
        test_labels = labels[22000:]

    print("Train Data:", train_data.shape)
    print("Train Label:", train_labels.shape)
    print("Test Data:", test_data.shape)
    print("Train Label:", test_labels.shape)

    trained_model, history = mt.train_model(
        model, train_data, train_labels, test_data, test_labels, epoch=7)

    if (save):
        if (mix):
            mt.save_model(trained_model, "aug_colour_model")
        else:
            mt.save_model(trained_model, "colour_model")
    if (test):
        mt.test_model(trained_model, test_data, test_labels)
    if (plot):
        mt.plot_history(history)
