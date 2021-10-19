import tempfile
from pathlib import Path
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import cog
from frozen_model.image_translator import ImageTranslator


class Predictor(cog.Predictor):
    def setup(self):
        args = parse_arguments()
        args.model = 'SPatchGAN_selfie2anime_scale3_cyc20_20210831.pb'
        size = 256
        self.translator = ImageTranslator(model_path=args.model,
                                          size=size,
                                          n_threads_intra=args.n_threads_intra,
                                          n_threads_inter=args.n_threads_inter)

    @cog.input(
        "image",
        type=Path,
        help="input image, model will generate female anime, support .png, .jpg and .jpeg",
    )
    def predict(self, image):
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        output = self.translator.translate(img)
        cv2.imwrite(str(out_path), output)
        return out_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file path or directory.')
    parser.add_argument('--model', type=str, help='.pb model path.')
    parser.add_argument('--n_threads_inter', type=int, default=1, help='Number of inter op threads.')
    parser.add_argument('--n_threads_intra', type=int, default=1, help='Number of intra op threads.')
    parser.add_argument('--n_iters', type=int, default=1)
    return parser.parse_args('')


class ImageTranslator:
    def __init__(self, model_path, size, n_threads_intra=1, n_threads_inter=1):
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=n_threads_intra,
                                inter_op_parallelism_threads=n_threads_inter)
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        config.gpu_options.allow_growth = True
        self._size = size
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(config=config, graph=self._graph)

        self._pb_file_path = model_path
        self._restore_from_pb()
        self._input_op = self._graph.get_tensor_by_name('test_domain_A:0')
        self._output_op = self._graph.get_tensor_by_name('test_fake_B:0')

    def _restore_from_pb(self):
        with self._graph.as_default():
            with gfile.FastGFile(self._pb_file_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
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
