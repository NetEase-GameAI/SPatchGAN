import tensorflow as tf


class ImageData:
    def __init__(self, load_size, augment_flag):
        self._load_size = load_size
        self._augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=3)

        if self._augment_flag is None:
            img = tf.image.resize_images(x_decode, [self._load_size, self._load_size])
        elif self._augment_flag == 'pad_crop':
            img = _augmentation_pad_crop(x_decode, self._load_size)
        elif self._augment_flag == 'resize_crop':
            img = _augmentation_resize_crop(x_decode, self._load_size)
        else:
            raise ValueError('Invalid augment_flag!')

        img = tf.cast(img, tf.float32) / 127.5 - 1
        return img


def _augmentation_pad_crop(image, size_out):
    image = tf.image.resize(image, [size_out, size_out])
    image = tf.cast(image, tf.uint8)
    # The shape info will be lost after random jpeg quality.
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=50, max_jpeg_quality=100)
    image = tf.reshape(image, [size_out, size_out, 3])

    pad_size = round(size_out * 0.05)
    # White padding
    image = tf.pad(image, paddings=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]], constant_values=255)
    image = tf.random_crop(image, [size_out, size_out, 3])
    image = _augmentation_general(image)
    return image


def _augmentation_resize_crop(image, size_out):
    aug_rand = tf.random_uniform([])
    image = tf.cond(aug_rand < 0.5,
                    lambda: _ugatit_resize_crop(image, size_out),
                    lambda: tf.image.resize(image, [size_out, size_out]))
    image = tf.cast(image, tf.uint8)
    # The shape info will be lost after random jpeg quality.
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=50, max_jpeg_quality=100)
    image = tf.reshape(image, [size_out, size_out, 3])
    image = _augmentation_general(image)
    return image


def _augmentation_general(image):
    # Operations that preserve the shape and are safe for most images.
    # These color changes should be done after padding to apply the changes on the paddings.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.02)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image


def _ugatit_resize_crop(image, size_out):
    augment_size = size_out
    if size_out == 256:
        augment_size += 30
    elif size_out == 512:
        augment_size += 60
    else:
        # Generalize the augmentation strategy in U-GAT-IT
        augment_size += round(augment_size * 0.1)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, [size_out, size_out, 3])
    return image
