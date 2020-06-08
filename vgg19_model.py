import inputs
import model_tools as mt
import cv2
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from keras import layers
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from tqdm import tqdm


def imagenet():
    pretrained = VGG19(include_top=False, weights="imagenet")
    # remove dense layers
    print(pretrained.summary())
    mt.save_model(pretrained, "imagenetVGG19")


def vgg_conv():
    images, labels = zip(*inputs.upload(img_size=(224, 224), dir="rawdata"))
    images = np.array(list(images))
    labels = np.array(list(labels))

    vggmodel = mt.load_model("imagenetVGG19")
    conv_data = []

    for img in tqdm(images):  # tqdm
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb[np.newaxis, ]
        img_rgb = preprocess_input(img_rgb)
        conv = vggmodel.predict_on_batch(img_rgb)
        conv = conv[0]
        conv_data.append(conv)

    conv_data = np.array(conv_data)
    print(conv_data.shape)
    with open(f"files/VGGCompressedData.npz", "wb") as file:
        np.savez_compressed(file, images=conv_data, labels=labels)


def get_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(7, 7, 512)),
        layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.3),
        layers.Dense(34, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='Adagrad',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
    return model


def conf_matrix(model, convs, labels):
    predictions = model.predict(convs)
    predictions = np.argmax(predictions, axis=1)

    df = pd.DataFrame({"Actual": labels, "Predicted": predictions}, columns=[
                      "Actual", "Predicted"])
    conf_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=[
        'Actual'], colnames=['Predicted'])
    sn.heatmap(conf_matrix, annot=True, fmt="d",
               cmap="YlGnBu", linewidths=.5, vmax=40)
    plt.show()


def grid(filename, model="topVGG19model", plot=False):
    img = cv2.imread(
        f"files/largescales/{filename}", cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (plot):
        fig, ax = plt.subplots()
        ax.imshow(img)

    vggconv = mt.load_model("imagenetVGG19")
    conv_data = []

    for i in tqdm(range(0, img.shape[0]-224, 224)):
        for j in range(0, img.shape[1]-224, 224):
            if (plot):
                rect = ax.add_patch(plt.Rectangle((i, j), 224, 224, edgecolor='red',
                                                  alpha=0.3, lw=2, facecolor='none'))
            crop = img[i:i+224, j:j+224]
            crop = crop[np.newaxis, ]
            crop = preprocess_input(crop)
            conv = vggconv.predict_on_batch(crop)
            conv_data.append(conv[0])
    conv_data = np.array(conv_data)

    vggmodel = mt.load_model(model)
    predictions = vggmodel.predict(conv_data)
    predictions = np.argmax(predictions, axis=1)

    predictions = np.bincount(predictions)/predictions.shape[0]*100
    landuses = [landuse for landuse in inputs.get_classes()]
    for i in range(34):
        try:
            pred = predictions[i]
        except:
            pred = 0.0
        print(f"{landuses[i]}: {pred}")
    if (plot):
        plt.show()


def run(model, plot=False, test=False, save=False, conf_matrix=False):
    convs, labels = inputs.load("VGGCompressedData")

    indices = np.arange(convs.shape[0])
    np.random.seed(0)
    np.random.shuffle(indices)

    convs = convs[indices]
    labels = labels[indices]

    train_data = convs[:23000]
    train_labels = labels[:23000]
    test_data = convs[23000:]
    test_labels = labels[23000:]

    print("Train Data:", train_data.shape)
    print("Train Label:", train_labels.shape)
    print("Test Data:", test_data.shape)
    print("Train Label:", test_labels.shape)

    trained_model, history = mt.train_model(
        model, train_data, train_labels, test_data, test_labels, epoch=15)
    if (save):
        mt.save_model(trained_model, "topVGG19model")
    if (test):
        mt.test_model(trained_model, test_data, test_labels)
    if (plot):
        mt.plot_history(history)
    if (conf_matrix):
        conf_matrix(trained_model, test_data, test_labels)
