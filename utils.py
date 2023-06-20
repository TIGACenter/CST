import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3, ResNet50


def list_file_paths(directory, extension=[".tif", ".png"]):
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


def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel/tf.reduce_sum(gauss_kernel)
    return gauss_kernel


def create_thesis_model(tile_size=128, channels=3, pretrained_path=None, final_layer_node=1, arq="resnet"):
    """
      - tile_size: (int) size of input tile.
      - pretrained_path: (str) loads model according to path.
      - final_layer_node: (int) number of nodes in final layer. If ==1, then final layer
                          is binary and has a sigmoid activation. If >1, then its multi-class
                          with a softmax activation.
    """
    if pretrained_path is None:
        model = tf.keras.models.Sequential()
        if arq == "resnet":
            base = ResNet50(weights='imagenet', include_top=False, input_shape=(tile_size, tile_size, channels))
        else:
            base = InceptionV3(weights='imagenet', include_top=False, input_shape=(tile_size, tile_size, channels))
        model.add(base)
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        if final_layer_node > 1:
            model.add(layers.Dense(final_layer_node, activation='softmax'))
        else:
            model.add(layers.Dense(final_layer_node, activation='sigmoid'))
    else:
        model = tf.keras.models.load_model(pretrained_path)
    return model