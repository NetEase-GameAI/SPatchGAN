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


def summary_by_keywords(keywords, node_type='tensor'):
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
