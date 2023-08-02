import time
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.engine import compile_utils

from .distortion_layers import tiga_rescale_layer


def st_loss(y_pred, y_true, y_dists, loss, st_loss, alpha=1, binary=False):
    l_0 = loss(y_true, y_pred)
    l_stab = sum([st_loss(y_pred, y_d) for y_d in y_dists])

    if binary:
        l_stab += sum([st_loss(1 - y_pred, 1 - y_d) for y_d in y_dists])

    return l_stab * alpha + l_0


def cst_metric(y_true, y_pred, base_metric, st_metric, alpha=1, binary=False):
    y_pred = tf.unstack(y_pred)  # y_pred is a tf tensor.
    # y_p = tf.squeeze(y_pred[0])  # y_pred[0] is the prediction on originals,
    # y_d = [tf.squeeze(i) for i in y_pred[1:]]  # y_pred[1:] is prediction on distorted
    y_p = y_pred[0]  # y_pred[0] is the prediction on originals,
    y_d = [i for i in y_pred[1:]]  # y_pred[1:] is prediction on distorted

    l_0 = base_metric(y_true, y_p)
    l_stab = sum([st_metric(y_p, d) for d in y_d])
    if binary:
        l_stab += sum([st_metric(1 - y_p, 1 - d) for d in y_d])
    return l_stab * alpha + l_0


class CSTMetric(tf.keras.metrics.MeanMetricWrapper):
    def __init__(
            self,
            base_metric,
            st_metric=tf.keras.metrics.kl_divergence,
            alpha=1,
            binary=False,
            name="cst",
            **kwargs
    ):
        super(CSTMetric, self).__init__(fn=cst_metric, name=name, base_metric=base_metric,
                                        st_metric=st_metric, alpha=alpha, binary=binary, **kwargs)


class CSTModel(tf.keras.Model):
    """
    Class for a model to be trained with Contrastive Stability Training (CST). Works similarly to
    a model object (tf.keras.Model(inputs=, outputs=)), with the addition of attributes
    to apply CST.

    Attributes:
        - dist_layers: (tf.keras.Layer or Sequential) Layer or Sequential object to generate
                       distorted images from input images
        - preprocessing_layer: (tf.keras.Layer or Sequential) Layer or Sequential object to preprocess input
                               images
        - rescale_layer: (tf.keras.Layer object) Layer to rescale the input image and distorted image
                         before passing to the network.
        - alpha: (float) weight of the stability component
        - n_st_components: (int) number of distorted images to generate for CST
        - binary: (bool) if True, the loss function is calculated for a binary problem; otherwise,
                  for a multiclass problem.
    """
    def __init__(
            self,
            dist_layers,
            preprocessing_layer=None,
            rescale_layer=None,
            alpha=1,
            n_st_components=0,
            binary=False,
            save_images=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.n_st_components = n_st_components
        self.binary = binary
        if rescale_layer is None:
            self.rescale_layer = tf.keras.layers.Lambda(lambda x: x)
        elif rescale_layer == 'default':
            self.rescale_layer = tiga_rescale_layer()
        else:
            self.rescale_layer = rescale_layer

        if preprocessing_layer is None:
            self.preprocessing_layer = tf.keras.layers.Lambda(lambda x: x)
        else:
            self.preprocessing_layer = preprocessing_layer

        if isinstance(dist_layers, list):
            self.dist_layers = dist_layers
            self.n_st_components = len(self.dist_layers)
        else:
            self.dist_layers = [dist_layers for _ in range(self.n_st_components)]
        self.save_images = save_images

    def compile(self, **kwargs):
        super(CSTModel, self).compile(**kwargs)
        self.st_loss_fn = st_loss  # loss function
        self.st_loss_tracker = tf.keras.metrics.Mean(name="loss")  # loss tracker for display during training
        self.metrics_dists = [
            compile_utils.MetricsContainer([type(m)(name=m.name + "_" + str(n)) for m in kwargs['metrics']])
            for n in range(self.n_st_components)
        ]
        self.metrics_i = compile_utils.MetricsContainer([type(m)(name=m.name) for m in kwargs['metrics']])
        self.cst_metric = CSTMetric(
            base_metric=type(kwargs['metrics'][0])(),
            alpha=self.alpha,
            binary=self.binary,
            name="cst",
        )

    # def _forward_pass(self, x, y, training=True):
    #     with tf.GradientTape() as tape:
    #         x_dists = [dist_layer(x, training=training) for dist_layer in self.dist_layers]
    #         x_dists = [self.preprocessing_layer(x_dist) for x_dist in x_dists]
    #         x_dists = [self.rescale_layer(x_dist) for x_dist in x_dists]
    #         y_dists = [self.call(x_dist, training=training) for x_dist in x_dists]
    #
    #         x = self.preprocessing_layer(x)
    #         x = self.rescale_layer(x)
    #         y_pred = self.call(x, training=training)
    #
    #         loss = self.st_loss_fn(
    #             y_pred=y_pred,
    #             y_true=y,
    #             y_dists=y_dists,
    #             binary=self.binary,
    #             loss=self.compiled_loss,  # TODO to compile
    #             st_loss=tf.keras.losses.kullback_leibler_divergence,  # TODO pasar a compile
    #             alpha=self.alpha  # TODO pasar a compile
    #         )
    #     return y_pred, y_dists, loss, tape

    def _update_states(self, y, y_pred, y_dists, loss, sample_weight):
        """
        Updates states of loss and metrics and returns verbose shown during training
        """

        self.st_loss_tracker.update_state(loss)
        # self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        self.metrics_i.update_state(y, y_pred, sample_weight=sample_weight)
        for i, dist in enumerate(self.metrics_dists):
            dist.update_state(y, y_dists[i], sample_weight=sample_weight)

        stacked_pred = tf.concat([y_pred[tf.newaxis, ...], y_dists], axis=0)
        self.cst_metric.update_state(y, stacked_pred, sample_weight=sample_weight)

        m_l_mean = {'loss_0': self.st_loss_tracker.result(), "cst_metric": self.cst_metric.result()}
        m_y = {m.name + "_": m.result() for m in self.metrics_i._metrics[0]}
        m_y_dist = {m.name: m.result() for d in self.metrics_dists for m in d._metrics[0]}
        return {**m_l_mean, **m_y, **m_y_dist}

    def train_step(self, data):
        # https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        # unpack data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            x_dists = [dist_layer(x, training=True) for dist_layer in self.dist_layers]
            x_dists = [self.preprocessing_layer(x_dist) for x_dist in x_dists]
            x_dists = [self.rescale_layer(x_dist) for x_dist in x_dists]
            y_dists = [self(x_dist, training=True) for x_dist in x_dists]

            x = self.preprocessing_layer(x)
            x = self.rescale_layer(x)
            y_pred = self(x, training=True)

            if self.run_eagerly and self.save_images:
                x_np = x.numpy()
                x_dists_np = [x_dist.numpy() for x_dist in x_dists]
                y_pred_np = y_pred.numpy()
                y_np = y.numpy()
                y_dists_np = [y_dist.numpy() for y_dist in y_dists]
                for i in range(x_np.shape[0]):
                    name = int(time.time() * 1000)
                    tf.keras.utils.save_img(f"{name}.png", x_np[i])
                    tf.keras.utils.save_img(f"{name}_y.png", y_pred_np[i] * 255)
                    tf.keras.utils.save_img(f"{name}_true.png", y_np[i])
                    for j in range(len(x_dists_np)):
                        tf.keras.utils.save_img(f"{name}_dist{j}.png", x_dists_np[j][i])
                        tf.keras.utils.save_img(f"{name}_y_dist{j}.png", y_dists_np[j][i])

            loss = self.st_loss_fn(
                y_pred=y_pred,
                y_true=y,
                y_dists=y_dists,
                binary=self.binary,
                loss=self.compiled_loss,  # TODO to compile
                st_loss=tf.keras.losses.KLDivergence(), # TODO pasar a compile
                alpha=self.alpha # TODO pasar a compile
            )

        # y_pred, y_dists, loss, tape = self._forward_pass(x, y, training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # self.st_loss_tracker.update_state(loss)
        # # self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # self.metrics_i.update_state(y, y_pred, sample_weight=sample_weight)
        # for i, dist in enumerate(self.metrics_dists):
        #     dist.update_state(y, y_dists[i], sample_weight=sample_weight)
        #
        # stacked_pred = tf.concat([y_pred[tf.newaxis, ...], y_dists], axis=0)
        # self.cst_metric.update_state(y, stacked_pred, sample_weight=sample_weight)
        # m_l_mean = {'loss_0': self.st_loss_tracker.result(), "cst_metric": self.cst_metric.result()}
        # # m_l_mean = {'loss_0': self.st_loss_tracker.result()}
        # m_y = {m.name + "_": m.result() for m in self.metrics_i._metrics[0]}
        # m_y_dist = {m.name: m.result() for d in self.metrics_dists for m in d._metrics[0] }
        # return {**m_l_mean, **m_y, **m_y_dist}

        verbose_metrics = self._update_states(y, y_pred, y_dists, loss, sample_weight)
        return verbose_metrics



    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape():
            x_dists = [dist_layer(x, training=None) for dist_layer in self.dist_layers]
            x_dists = [self.preprocessing_layer(x_dist) for x_dist in x_dists]
            x_dists = [self.rescale_layer(x_dist) for x_dist in x_dists]
            y_dists = [self(x_dist, training=False) for x_dist in x_dists]

            x = self.preprocessing_layer(x)
            x = self.rescale_layer(x)
            y_pred = self(x, training=False)

            loss = self.st_loss_fn(
                y_pred=y_pred,
                y_true=y,
                y_dists=y_dists,
                binary=self.binary,
                loss=self.compiled_loss,  # TODO to compile
                st_loss=tf.keras.losses.kullback_leibler_divergence,  # TODO pasar a compile
                alpha=self.alpha  # TODO pasar a compile
            )
        # _pred, y_dists, loss = self._forward_pass(x, y, training=None)


        # self.metrics_i.update_state(y, y_pred, sample_weight=sample_weight)
        # return {m.name: m.result() for m in self.metrics_i._metrics[0]}
        verbose_metrics = self._update_states(y, y_pred, y_dists, loss, sample_weight)
        return verbose_metrics