import os
import requests
import shutil
import pathlib
import zipfile
import tarfile
import argparse
import tensorflow as tf
import numpy as np
import cv2

from . import utils


def load_aj_idc(PATH=None):
    """
    Dataset: Invasive Ductal Carcinoma Identification.
    Author: Andrew Janowczyk
    Source: http://andrewjanowczyk.com/deep-learning/
    Paper: Janowczyk, A., & Madabhushi, A. (2016). Deep learning for digital pathology
           image analysis: A comprehensive tutorial with selected use cases. Journal of
           pathology informatics, 7(1), 29. https://doi.org/10.4103/2153-3539.186902
    """

    URL = "http://andrewjanowczyk.com/wp-static/IDC_regular_ps50_idx5.zip"
    if PATH is None:
        PATH = "data/aj"

    print("Dataset: Invasive Ductal Carcinoma Identification.\nAuthor: Andrew Janowczyk")
    print("source: http://andrewjanowczyk.com/deep-learning/")

    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, stream=True)
    zip_name = os.path.join(PATH, os.path.basename(URL))
    dir_name = os.path.splitext(zip_name)[0]

    print("Downloading dataset (might take a few minutes)...")
    with open(zip_name, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    with zipfile.ZipFile(zip_name, 'r') as f:
        f.extractall(dir_name)
    os.remove(zip_name)
    print("...Download complete")

    print("Organizing images into class folders...")
    pathlib.Path(dir_name + "/0").mkdir(parents=True, exist_ok=True)
    pathlib.Path(dir_name + "/1").mkdir(parents=True, exist_ok=True)
    paths = utils.list_file_paths(dir_name, [".png"])
    for path in paths:
        to_delete = path.split(os.path.sep)[-3] + "/"
        shutil.move(path, path.replace(to_delete, ""))

    for folder in list(os.walk(dir_name))[1:]:
        # folder example: ('FOLDER/3', [], ['file'])
        if os.path.basename(folder[0]) not in ["0", "1"]:
            shutil.rmtree(folder[0])
    print("...Organization complete")


def load_cifar_10(PATH=None):
    if PATH is None:
        PATH = "data/cifar-10"

    print("Dataset: CIFAR-10")
    print("Source: https://www.cs.toronto.edu/~kriz/cifar.html")

    print("Downloading dataset (might take a few minutes)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    for i in range(len(x_train)):
        img_path = f"{PATH}/train/{y_train[i]}"
        pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(img_path, f"{i:05d}.png"),
                    cv2.cvtColor(x_train[i], cv2.COLOR_RGB2BGR))

    for i in range(len(x_test)):
        img_path = f"{PATH}/test/{y_test[i]}"
        pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(img_path, f"{i:05d}.png"),
                    cv2.cvtColor(x_test[i], cv2.COLOR_RGB2BGR))
    print("...Download complete")


def load_cifar_10_c():
    """
    Dataset: CIFAR-10 Corrupted.
    Author: Dan Hendrycks and Thomas Dietterich
    Source: https://zenodo.org/record/2535967#.Yr8BD79BzmE
    Paper: Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network
           robustness to common corruptions and perturbations. arXiv preprint
           arXiv:1903.12261. https://doi.org/10.48550/arXiv.1903.12261
    """

    URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    PATH = "data"

    print("Dataset: CIFAR-10 Corrupted")
    print("Source: https://zenodo.org/record/2535967#.Yr8BD79BzmE")

    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, stream=True)
    tar_name = os.path.join(PATH, os.path.basename(URL))
    dir_name = os.path.splitext(tar_name)[0]

    print("Downloading dataset (might take a few minutes)...")
    with open(tar_name, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    tar = tarfile.open(tar_name)
    tar.extractall(PATH)
    tar.close()
    os.remove(tar_name)
    print("...Download complete")

    print("Organizing images into class folders...")
    labels = np.load(os.path.join(dir_name, "labels.npy"))
    paths = utils.list_file_paths(dir_name, [".npy"])
    paths = [i for i in paths if "labels.npy" not in i]
    for p in paths:
        arr = np.load(p)
        for i in range(len(arr)):
            img_path = f"{os.path.splitext(p)[0]}/{labels[i]}"
            pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(img_path, f"{i:05d}.png"),
                        cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
        os.remove(p)
    print("...Organization complete")


if __name__ == "__main__":
    help = "Name of the dataset you want to download. Options: idc, cifar-10, cifar-10-c"
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', nargs='+',
                        help="Name of the dataset you want to download", type=str)
    args = parser.parse_args()
    d = False
    if "idc" in args.name:
        load_aj_idc()
        d = True
    if "cifar-10-c" in args.name:
        load_cifar_10_c()
        d = True
    if  "cifar-10" in args.name:
        load_cifar_10()
        d = True
    if not d: print("No valid dataset was provided. Run --help for extra info.")
