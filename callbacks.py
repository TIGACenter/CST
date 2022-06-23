import tensorflow as tf


class SaveModelOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, base_name, st=False):
        self.base_name = base_name
        self.st = st
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if self.st:
            self.model.layers[1].save(self.base_name + "_e" + str(epoch + 1) + ".h5")
        else:
            self.model.save(self.base_name + "_e" + str(epoch + 1) + ".h5")