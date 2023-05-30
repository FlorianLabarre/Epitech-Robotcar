"""
04-more-training
author: @maximellerbach
"""

import glob
import json
import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence


def load_image_and_json(img_path, json_path):
    """
    Load the image using img_path, load the json data using json_path,
    return both as a tuple (image, json_data)
    """
    img = cv2.imread(img_path)

    with open(os.path.normpath(json_path), "r") as json_file:
        json_data = json.load(json_file)

    return img, json_data


class DataGenerator(Sequence):
    def __init__(
        self,
        data_directory,
        transform_funcs,
        batch_size=32,
    ):
        """Init of the class.

        Args:
            data_directory (list): list or list of list of json_paths to train on.
            batch_size (int, optional): _description_. Defaults to 64.
            transform_funcs (list): list of data augm funcs of prototype: (image, label) -> (image, label)

        Raises:
            ValueError: directory should be non-empty

        Be carefull to check that the number of images and json files match !
        You can also check that the name of the images and json files match.
        """
        self.transform_funcs = transform_funcs
        self.batch_size = batch_size

        self.image_paths = glob.glob(os.path.join(data_directory, "*.png"))
        self.json_paths = glob.glob(os.path.join(data_directory, "*.json"))
        assert len(self.image_paths) > 0, "no images in directory were found"
        assert len(self.image_paths) == len(
            self.json_paths), f"number of X and Y missmatch {len(self.image_paths)} {len(self.json_paths)}"
        self.length = len(self.image_paths)

        # just check that every img / json paths does match
        for (img_p, json_p) in zip(self.image_paths, self.json_paths):
            img_name = img_p.split(os.path.sep)[-1].split(".png")[0]
            json_name = json_p.split(os.path.sep)[-1].split(".json")[0]

            assert img_name, json_name

    def __load_next(self):
        """Prepare a batch of data for training.

        X represents the input data, and Y the expected outputs (as in Y=f(X))

        Returns:
            tuple(list, list): X and Y.
        """

        X = []
        Y = []

        list = np.random.randint(0, self.length, size=self.batch_size)
        for i in list:
            img_path = self.image_paths[i]
            json_path = self.json_paths[i]
            image, data = load_image_and_json(img_path, json_path)

            for func in self.transform_funcs:
                image, data = func(image, data)
            X.append(image)
            Y.append(data["steering"])
        X = np.array(X) / 255
        Y = np.array(Y)

        return X, Y

    def __len__(self):
        return self.length // self.batch_size + 1

    def __getitem__(self, index):
        return self.__load_next()


"""
Now some transform / data augmentation functions, 
they should only activate some of the time, for example 50% for the flip.
To do so, you can use np.random.random() and check if it is below a certain threshold.
I know this is not the most elegant/performant way, but it is easy to read and understand.

you can get creative with those !
Here are two of them, you can use many more !
"""


def flip(image: np.ndarray, data: dict):
    """Simple image flip. Also flip the label."""

    rand = np.random.random()
    if rand < 0.5: # 50%
        data["steering"] = data["steering"] * -1
        image = cv2.flip(image, 1)

    return image, data


def noise(image: np.ndarray, data: dict, mult=10):
    """Add some noise to the image."""
    
    rand = np.random.random()
    if rand < 0.1: # 10%
        noise = np.random.randint(-mult, mult, dtype=np.int8)
        image = image + noise # not perfect, here there could be some unsigned overflow 

    return image, data
