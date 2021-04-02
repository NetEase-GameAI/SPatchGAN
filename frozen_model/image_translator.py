import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow.python.platform import gfile

curPath = os.path.abspath(os.path.dirname(__file__))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ImageTranslator:
    def __init__(self, model_path, size, n_threads_intra=1, n_threads_inter=1):
        config = tf.ConfigProto(intra_op_parallelism_threads=n_threads_intra,
                                inter_op_parallelism_threads=n_threads_inter)
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        config.gpu_options.allow_growth = True
        self._size = size
        self._graph = tf.Graph()
        self._sess = tf.Session(config=config, graph=self._graph)

        self._pb_file_path = model_path
        self._restore_from_pb()
        self._input_op = self._graph.get_tensor_by_name('test_domain_A:0')
        self._output_op = self._graph.get_tensor_by_name('test_fake_B:0')

    def _restore_from_pb(self):
        with self._graph.as_default():
            with gfile.FastGFile(self._pb_file_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

    def _input_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self._size, self._size))
        img = np.expand_dims(img, axis=0)
        image_input = img / 127.5 - 1
        return image_input

    @staticmethod
    def _output_transform(output):
        output = ((output + 1.) / 2) * 255.0
        image_output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_RGB2BGR)
        return image_output

    def translate(self, image):
        """ Translate an image from the source domain to the target domain"""
        image_input = self._input_transform(image)
        output = self._sess.run(self._output_op, feed_dict={self._input_op: image_input})[0]
        return self._output_transform(output)
