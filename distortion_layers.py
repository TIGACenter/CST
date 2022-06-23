import tensorflow as tf
import tensorflow_probability as tfp


def create_classifier_top(width=128, classes=3):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(width, activation='relu'),
            tf.keras.layers.Dense(width, activation='relu'),
            tf.keras.layers.Dense(classes, activation='softmax')
        ],
        name="classifier_top"
    )


def tiga_rescale_layer():
    # return tf.keras.layers.Lambda(lambda x: ((x / 255.) - .5) * 2)
    return tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)


def tiga_unrescale_layer():
    # return tf.keras.layers.Lambda(lambda x: ((x * .5) + 0.5) * 255.)
    return tf.keras.layers.Rescaling(scale=127.5, offset=127.5)


class DistortionLayer(tf.keras.layers.Layer):
    """
    Utility layer that concatenates all distortion layers into one

    Attributes:
        - layers: (list(layer)) list of layer instances to stack as a Sequential model
    """
    def __init__(self, layers=[]):
        super(DistortionLayer, self).__init__()
        self.layers = layers

    def call(self, inputs, training=None):
        if training or training is None:
            if len(self.layers) > 0:
                x = self.layers[0](inputs)
                for layer in self.layers[1:]:
                    x = layer(x)
                return x
        return inputs


class RandomColorByChannel(tf.keras.layers.Layer):
    """
    Layer to apply a random color distortion to input image. A random factor is added to each of
    the RGB channels of the image or tensor. Image must be RGB [0,255] (int or float).
    Outputs image [0,255] (float).

    Attributes:
        - factor: (list(float, float, float) or tuple(float, float, float)) max factors for each
                  RGB channel. For each channel, a random factor between the ranges
                  [-factor[0], factor[0]], [-factor[1], factor[1]] and [-factor[2], factor[2]]
                  is added to R, G and B respectively.

    """
    def __init__(self, factor=[0,0,0]):
        super(RandomColorByChannel, self).__init__()
        self.factor = tf.constant(factor, dtype=tf.float32)

    def call(self, inputs, training=None):
        if training or training is None:
            batch_size = tf.shape(inputs)[0]
            to_add = tf.random.uniform(shape=[batch_size,1,1,3], minval=-1, maxval=1, dtype=tf.float32)
            to_add *= self.factor
            return tf.clip_by_value(tf.add(tf.cast(inputs, dtype=tf.float32), to_add), 0, 255)
        return inputs


class ClipByValue(tf.keras.layers.Layer):
    """
    Layer to clip input image to range [0,255]. Required for distortions where pixel values go beyond
    that range. E.g. tf.layers.RandomContrast may show that behaviour if input is not type int.

    Image must be RGB [0,255] (int or float). Outputs image [0,255] (float).
    """
    def __init__(self):
        super(ClipByValue, self).__init__()

    def call(self, inputs):
        return tf.clip_by_value(inputs, 0, 255)


class RandomSaturation(tf.keras.layers.Layer):
    """
    Layer to apply a random change in saturation to the image. Uses tf.image.random_saturation,
    which converts image to HSV, adjusts the S channel by a factor and changes back to RGB.
    A factor > 1 makes the image more saturated (i.e. stronger color), and < 1 makes it less
    saturated (i.e. grayer).

    Image must be RGB [0,255] (int or float). Outputs image [0,255] (float).

    Attributes:
        - lower: (float) min factor for saturation. has to be > 0.
        - upper: (float) max factor for saturation. has to be > lower
    """
    def __init__(self, lower=1., upper=1.0001):
        super(RandomSaturation, self).__init__()
        self.lower = lower
        self.upper = upper

    def call(self,inputs, training=None):
        if training or training is None:
            return tf.image.random_saturation(inputs, self.lower, self.upper)
        return inputs


class RandomGaussianBlur(tf.keras.layers.Layer):
    """
    Layer to apply gaussian blur to image. A gaussian kernel is convolved to the input image,
    resulting in a blurred image.

    Image must be RGB [0,255] (int or float). Outputs image [0,255] (float). Input must be of
    shape [batch_size, height, width, channels].

    Attributes
        - filter_shape: (int) level of the gaussian kernel as a square. size = level * 2 + 1
        - sigma: (float) standard deviation of the gaussian kernel distribution.
        - interpolation: (str) interpolation applied for resizing
    """

    def __init__(self, filter_shape=1, sigma=1., interpolation='bilinear'):
        super(RandomGaussianBlur, self).__init__()
        self.filter_shape = filter_shape
        self.sigma = sigma
        self.interpolation = interpolation
        self.resize = None

    def call(self, inputs, training=None):
        if training or training is None:
            inputs = tf.cast(inputs, dtype=tf.float32)
            input_shape = tf.shape(inputs)
            sigma = tf.random.uniform(shape=[], minval=0, maxval=self.sigma, dtype=tf.float32)
            if self.resize is None:
                self.resize = tf.keras.layers.Resizing(
                    height=input_shape[-3], width=input_shape[-2], interpolation=self.interpolation)
            # adaptation from https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319

            gauss_kernel = self.gaussian_kernel(size=self.filter_shape, mean=0., std=sigma + .00001)
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            tf_image_B = tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, 1])
            tf_image_G = tf.slice(inputs, [0, 0, 0, 1], [-1, -1, -1, 1])
            tf_image_R = tf.slice(inputs, [0, 0, 0, 2], [-1, -1, -1, 1])
            strides = tf.ones(tf.size(tf.shape(inputs)))
            tf_image_B = tf.nn.conv2d(tf_image_B, gauss_kernel, strides=[1,1,1,1], padding="SAME")
            tf_image_G = tf.nn.conv2d(tf_image_G, gauss_kernel, strides=[1,1,1,1], padding="SAME")
            tf_image_R = tf.nn.conv2d(tf_image_R, gauss_kernel, strides=[1,1,1,1], padding="SAME")
            tf_image_B = tf.squeeze(tf_image_B)
            tf_image_G = tf.squeeze(tf_image_G)
            tf_image_R = tf.squeeze(tf_image_R)
            x = tf.stack([tf_image_B, tf_image_G, tf_image_R], axis=-1)

            # return self.resize(x)
            return x
        return inputs


    @staticmethod
    def gaussian_kernel(size: int, mean: float, std: float):
        """Makes 2D gaussian Kernel for convolution."""
        d = tfp.distributions.Normal(mean, std)
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        return gauss_kernel


class GaussianNoise(tf.keras.layers.Layer):
    """
    Layer to add gaussian noise to inputs.

    Image must be RGB [0,255] (int or float). Outputs image [0,255] (float).

    Attributes:
        mean: (float) mean for the normal distribution of the pixel value noise.
        std: (float) standard deviation for the normal distribution of the pixel value noise.
    """
    def __init__(self, mean=0., std=.0):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def call(self, inputs, training=None):
        if training or training is None:
            inputs = tf.cast(inputs, dtype=tf.float32)
            std = tf.random.uniform(shape=[], minval=0, maxval=self.std)
            noise = tf.random.normal(shape=tf.shape(inputs), mean=self.mean, stddev=std, dtype=tf.float32)
            return tf.clip_by_value(tf.add(inputs, noise), 0, 255)
        return inputs


class RandomCropLayer(tf.keras.layers.Layer):
    """
    Layer to crop a random section of an image. A random translation is first applied on the image,
    followed by a zoom and resizing.

    Attributes:
        - min_height: (float) minimum height as a fraction of 1, with 1 being the whole height.
#       - min_width: (float) minimum width as a fraction of 1, with 1 being the whole width.
#       - max_height: (float) maximum height as a fraction of 1, with 1 being the whole height.
#       - max_width: (float) maximum width as a fraction of 1, with 1 being the whole width.
    """
    def __init__(self, min_height=1., min_width=1., max_height=1., max_width=1.):
        super(RandomCropLayer, self).__init__()
        self.min_height = min_height
        self.min_width = min_width
        self.max_height = max_height
        self.max_width = max_width
        self.random_translation = tf.keras.layers.RandomTranslation(
            (1 - self.min_height) / 2, (1 - self.min_width) / 2, interpolation='nearest'
        )
        self.random_zoom = tf.keras.layers.RandomZoom(
            (-(1 - self.min_height), -(1 - self.max_height)), (-(1 - self.min_width), -(1 - self.max_width))
        )

    def call(self, inputs, training=None):
        if training or training is None:
            x = self.random_translation(inputs)
            return self.random_zoom(x)
        return inputs