"""
04-more-training
author: @maximellerbach
"""

import glob
import os

import cv2
import numpy as np
import tensorflow as tf
# also imports your data-augmentation functions here
from utils import DataGenerator, flip, noise

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')


def build_model() -> tf.keras.Model:
    """
    Build a simple CNN model: stack some layers and compile it, please mind the size of the model.
    It might also be a good idea to try to reduce as much as possible the number of operations !
    a quick reminder: https://www.tensorflow.org/api_docs/python/tf/keras/Model

    Be careful with the input shape (120, 160, 3).
    Note that we want to use the activation function "tanh" on the output layer.

    If you are brave enough, you can try to add more inputs/outputs BUT you will also have to change the DataGenerator class to match those.

    For the model compilation, use adam optimizer and mse loss.
    """
    inputs = tf.keras.Input(shape=(120,160,3))
    x = tf.keras.layers.Conv2D(16, 3, 2, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, 2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(48, 3, 2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, 2, activation="relu")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(2, activation="tanh")(x)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mse"]
    )
    return model


def train_model(model: tf.keras.Model, datagen: DataGenerator, epochs=10):
    """
    Provide the datagen object to the fit function as source of data.

    (Bonus) It is also a good idea to use a test_datagen to monitor the performance of the model on unseen data.
    In order to do that, you would have to create a new DataGenerator object with a different data directory.
    """
    model.fit(
        datagen,
        steps_per_epoch=len(datagen),
        epochs=epochs,
    )


def load_model(model_path):
    """"Simply load the model."""
    return tf.keras.models.load_model(model_path)


def predict(model, data_path):
    """
    Predict the steering angle for each image in the data_path.
    You can sort the images by name (date) to get the correct order then play the images as a video.

    hint: you can use cv2 to display the images
    You can also draw a visualisation of the steering angle on the image.
    """

    def sort_func(x):
        return float(x.split(os.path.sep)[-1].split(".")[0])

    img_paths = sorted(glob.glob(os.path.join(
        data_path, "*.png")), key=sort_func)

    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (160, 120))
        x = img / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model(x)[0][0]

        # draw horizontal line
        cv2.line(img, (80, 110), (int(80 + pred * 40), 110), (0, 0, 255), 4)

        # display image
        cv2.imshow("img", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    # model = load_model("trained_model.h5")
    model = build_model()
    model.summary()

    # if you have not implemented any transform funcs yet, just put an empty list []
    datagen = DataGenerator(data_path, [flip, noise], batch_size=32)

    # if the traning takes too much time, you can try to reduce the batch_size and the number of epochs
    train_model(model, datagen, epochs=100)

    model.save("trained_model3.h5")
    # predict(model, data_path)
