import os
import numpy as np


def list_file_paths(directory, extension=[".tif", ".png"]):
    """
    lists all files in ´directory´ and its sub-folders with
    extensions in ´extension´
    """
    paths = []
    for path, _, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1] in extension or len(extension)==0:
                paths.append(os.path.join(path, name))
    return paths


def normalize_image(image):
    norm_image = image / 255.
    norm_image -= 0.5
    norm_image *= 2.
    return norm_image


def denormalize_image(image):
    denorm_image = image / 2.
    denorm_image += 0.5
    denorm_image *= 255.
    return denorm_image


def get_class_weights(classes):
    """
    calculates class weights for training
    inputs:
      - class_counts: (dict) dict containing input count per class
    outputs:
      - (dict) dict containing weights per class
    """

    cl, counts = np.unique(classes, return_counts=True)
    return {i: max(counts) / counts[cl[i]] for i in cl}
