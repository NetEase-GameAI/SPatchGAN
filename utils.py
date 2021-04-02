import cv2
import tensorflow as tf
import numpy as np
from glob import glob
from tensorflow.contrib import slim


def show_all_variables():
    """Show all variables in the graph."""
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_img_paths(input_dir: str, dataset_struct: str = 'plain'):
    """Get all images in an input directory."""
    exts = ['jpg', 'jpeg', 'png']
    imgs = []
    for ext in exts:
        if dataset_struct == 'plain':
            pattern = input_dir + '/*.{}'.format(ext)
        elif dataset_struct == 'tree':
            pattern = input_dir + '/*/*.{}'.format(ext)
        else:
            raise ValueError('Invalid dataset_struct!')
        imgs.extend(glob(pattern))
    return imgs


def get_img_paths_auto(input_dir: str):
    """Auto detect the directory structure and get all images."""
    dataset = get_img_paths(input_dir)
    if len(dataset) == 0:
        dataset = get_img_paths(input_dir, dataset_struct='tree')
    return dataset


def summary_by_keywords(keywords, node_type='tensor'):
    """Generate summary for the tf nodes whose names match the keywords."""
    summary_list = []
    if node_type == 'tensor':
        all_nodes = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
    elif node_type == 'variable':
        all_nodes = tf.trainable_variables()
    else:
        raise RuntimeError('Invalid target!')

    keyword_list = keywords if isinstance(keywords, list) else [keywords]

    # Include a node if its name contains all keywords.
    nodes = [node for node in all_nodes if all(keyword in node.name for keyword in keyword_list)]

    for node in nodes:
        summary_list.append(tf.summary.scalar(node.name + "_min", tf.reduce_min(node)))
        summary_list.append(tf.summary.scalar(node.name + "_max", tf.reduce_max(node)))
        node_mean = tf.reduce_mean(node)
        summary_list.append(tf.summary.scalar(node.name + "_mean", node_mean))
        # Calculate the uncorrected standard deviation
        node_stddev = tf.sqrt(tf.reduce_mean(tf.square(node - node_mean)))
        summary_list.append(tf.summary.scalar(node.name + "_stddev", node_stddev))

    return summary_list


def batch_resize(x, img_size):
    """Resize the NHWC Numpy images."""
    x_up = np.zeros((x.shape[0], img_size, img_size, 3))
    for i in range(x.shape[0]):
        x_up[i, :, :, :] = cv2.resize(x[i, :, :, :], dsize=(img_size, img_size))
    return x_up


def save_images(images, size, image_path):
    """Save a grid of images."""
    return _imsave(_inverse_transform(images), size, image_path)


def _inverse_transform(images):
    return ((images+1.) / 2) * 255.0


def _imsave(images, size, path):
    images = _merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)


def _merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img


def load_test_data(image_path, size=256):
    """Load test images with OpenCV."""
    img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(size, size))
    img = np.expand_dims(img, axis=0)
    img = img / 127.5 - 1

    return img
