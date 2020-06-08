import inputs
import tensorflow as tf
import pandas as pd
import matplotlib as mpl

configs = inputs.config()
if (configs["matplotlib_gui"]):
    mpl.use('tkagg')
else:
    mpl.use('agg')

import matplotlib.pyplot as plt  # noqa


def train_model(model, train_data, train_labels, test_data, test_labels, epoch=10):
    # TODO implement earlystop callback
    return model, model.fit(train_data,
                            train_labels,
                            epochs=epoch,
                            # batch_size=512,
                            validation_split=0.1,
                            verbose=2)


def load_model(name):
    model = tf.keras.models.load_model(
        f'files/{name}.h5', custom_objects={'softmax_v2': tf.nn.softmax})
    return model


def save_model(model, name):
    print("Saving model...")
    model.save(f'files/{name}.h5')
    print("Saved.")


def test_model(model, test_data, test_labels):
    test_loss, test_acc, a = model.evaluate(test_data, test_labels)
    print('Test accuracy:', test_acc)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
    plt.ylim([0, 2])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Acc')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Acc')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
