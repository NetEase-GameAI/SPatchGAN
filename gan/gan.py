import os
import tensorflow as tf
from datetime import datetime
import time
from utils import get_img_paths, save_images, load_test_data


class GAN:
    def __init__(self, model_name, sess, args):
        # General
        self._model_name = model_name
        self._sess = sess
        self._saver = None
        self._dataset_name = args.dataset
        self._test_dataset_name = args.test_dataset or args.dataset
        self._dataset_struct = args.dataset_struct
        self._suffix = args.suffix

        # Input
        self._img_size = args.img_size
        self._augment_type = args.augment_type
        trainA_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', self._dataset_name, 'trainA')
        trainB_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', self._dataset_name, 'trainB')
        self._trainA_dataset = get_img_paths(trainA_dir, self._dataset_struct)
        self._trainB_dataset = get_img_paths(trainB_dir, self._dataset_struct)
        self._dataset_num = max(len(self._trainA_dataset), len(self._trainB_dataset))

        # Generator
        self._gen = None

        # Directory
        model_dir = "{}_{}_{}".format(self._model_name, self._dataset_name, self._suffix)
        self._checkpoint_dir = os.path.join(args.output_dir, model_dir, args.checkpoint_dir)
        self._sample_dir = os.path.join(args.output_dir, model_dir, args.sample_dir)
        self._log_dir = os.path.join(args.output_dir, model_dir, args.log_dir)
        self._result_dir = os.path.join(args.output_dir, model_dir, args.result_dir)
        for dir_ in [self._checkpoint_dir, self._sample_dir, self._log_dir, self._result_dir]:
            os.makedirs(dir_, exist_ok=True)

        print()
        print('##### Information #####')
        print('Number of trainA/B images: {}/{}'.format(len(self._trainA_dataset), len(self._trainB_dataset)) )
        print()

    def build_model_train(self):
        pass

    def build_model_test(self):
        self._test_domain_a = tf.placeholder(tf.float32, [1, self._img_size, self._img_size, 3],
                                             name='test_domain_A')
        test_fake_b = self._gen.translate(self._test_domain_a, scope='gen_a2b')
        self._test_fake_b = tf.identity(test_fake_b, 'test_fake_B')

    def train(self):
        pass

    def test(self):
        tes_a_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', self._test_dataset_name, 'testA')
        test_a_files = get_img_paths(tes_a_dir, self._dataset_struct)

        if self._saver is None:
            self._saver = tf.train.Saver()
        could_load, checkpoint_counter = self._load_ckpt(self._checkpoint_dir)
        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")
            raise RuntimeError("Failed to load the checkpoint")

        dataset_tag = '' if self._test_dataset_name == self._dataset_name else self._test_dataset_name + '_'
        result_dir = os.path.join(self._result_dir, dataset_tag + str(checkpoint_counter))
        os.makedirs(result_dir, exist_ok=True)

        st = time.time()
        for sample_file in test_a_files:  # A -> B
            print('Processing source image: ' + sample_file)
            input = load_test_data(sample_file, size=self._img_size)
            fake_img = self._sess.run(self._test_fake_b, feed_dict={self._test_domain_a: input})

            if self._dataset_struct == 'plain':
                dst_dir = result_dir
            elif self._dataset_struct == 'tree':
                src_dir = os.path.dirname(sample_file)
                dirname_level1 = os.path.basename(src_dir)
                dirname_level2 = os.path.basename(os.path.dirname(src_dir))
                dst_dir = os.path.join(result_dir, dirname_level2, dirname_level1)
                os.makedirs(dst_dir, exist_ok=True)
            else:
                raise RuntimeError('Invalid dataset_type!')
            image_path = os.path.join(dst_dir, os.path.basename(sample_file))
            save_images(fake_img[[0],:], [1, 1], image_path)

        time_cost = time.time() - st
        time_cost_per_img_ms = round(time_cost * 1000 / len(test_a_files))
        print('Time cost per image: {} ms'.format(time_cost_per_img_ms))

    def freeze_graph(self):
        self._saver = tf.train.Saver()
        could_load, checkpoint_counter = self._load_ckpt(self._checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError("Failed to load the checkpoint")

        output_dir = os.path.join(self._checkpoint_dir, 'pb')
        os.makedirs(output_dir, exist_ok=True)
        time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_file = os.path.join(output_dir, 'output_graph_' + time_stamp + '.pb')

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=self._sess,
            input_graph_def=self._sess.graph_def,
            output_node_names=['test_fake_B'])

        # Save the frozen graph
        with open(output_file, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    def _save_ckpt(self, checkpoint_dir, step):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._saver.save(self._sess, os.path.join(checkpoint_dir, self._model_name + '.model'), global_step=step)

    def _load_ckpt(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self._saver.restore(self._sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

