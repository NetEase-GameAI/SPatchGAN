from glob import glob
import tensorflow as tf
from tensorflow.contrib import slim


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_img_paths(input_dir: str, level=1):
    exts = ['jpg', 'jpeg', 'png']
    imgs = []
    for ext in exts:
        if level == 1:
            pattern = input_dir + '/*.{}'.format(ext)
        elif level == 2:
            pattern = input_dir + '/*/*.{}'.format(ext)
        else:
            raise ValueError('Invalid level!')
        imgs.extend(glob(pattern))
    return imgs
