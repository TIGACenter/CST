import os
import tensorflow as tf
import pathlib


class EpochSaver(tf.keras.callbacks.Callback):
    def __init__(self, layer_name, model_path, base_name):
        self.layer_name = layer_name
        self.model_path = model_path
        self.base_name = base_name
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if not (epoch + 1) % 1 == 0:
            return
        pathlib.Path(self.model_path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(self.model_path, self.base_name) + "_e" + str(epoch + 1) + ".h5"
        self.model.get_layer(self.layer_name).save(path)
        print("class weights saved to path: ")
        print(path)