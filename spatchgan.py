import os
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
from utils import get_img_paths, summary_by_keywords, batch_resize, save_images, load_test_data
from ops import l1_loss, adv_loss, regularization_loss
from imagedata import ImageData
from discriminator.discriminator_spatch import DiscriminatorSPatch
from discriminator.discriminator_patch import DiscriminatorPatch
from generator.generator_basic_res import GeneratorBasicRes


class SPatchGAN:
    def __init__(self, sess, args):
        # General
        self._model_name = 'SPatchGAN'
        self._sess = sess
        self._saver = None
        self._dataset_name = args.dataset
        self._test_dataset_name = args.test_dataset or args.dataset
        self._dataset_struct = args.dataset_struct
        self._suffix = args.suffix

        # Training
        self._n_steps = args.n_steps
        self._n_iters_per_step = args.n_iters_per_step
        self._batch_size = args.batch_size
        self._img_save_freq = args.img_save_freq
        self._ckpt_save_freq = args.ckpt_save_freq
        self._summary_freq = args.summary_freq
        self._decay_step = args.decay_step
        self._init_lr = args.lr
        self._adv_weight = args.adv_weight
        self._reg_weight = args.reg_weight
        self._cyc_weight = args.cyc_weight
        self._id_weight = args.id_weight
        self._gan_type = args.gan_type

        # Input
        self._img_size = args.img_size
        self._augment_type = args.augment_type
        trainA_dir = os.path.join(os.path.dirname(__file__), 'dataset', self._dataset_name, 'trainA')
        trainB_dir = os.path.join(os.path.dirname(__file__), 'dataset', self._dataset_name, 'trainB')
        self._trainA_dataset = get_img_paths(trainA_dir, self._dataset_struct)
        self._trainB_dataset = get_img_paths(trainB_dir, self._dataset_struct)
        self._dataset_num = max(len(self._trainA_dataset), len(self._trainB_dataset))

        # Discriminator
        self._dis = self._create_dis(args)

        # Generator
        self._gen = self._create_gen(args)
        self._gen_bw = self._create_gen_bw(args)
        self._resolution_bw = self._img_size // args.resize_factor_gen_bw

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

    @staticmethod
    def _create_dis(args):
        if args.dis_type == 'spatch':
            stats = []
            if args.mean_dis:
                stats.append('mean')
            if args.max_dis:
                stats.append('max')
            if args.mean_dis:
                stats.append('stddev')
            return DiscriminatorSPatch(ch=args.ch_dis,
                                       n_downsample_init=args.n_downsample_init_dis,
                                       n_scales=args.n_scales_dis,
                                       n_adapt=args.n_adapt_dis,
                                       n_mix=args.n_mix_dis,
                                       logits_type=args.logits_type_dis,
                                       stats=stats,
                                       sn=args.sn_dis)
        elif args.dis_type == 'patch':
            return DiscriminatorPatch(ch=args.ch_dis,
                                      n_downsample_init=args.n_downsample_init_dis,
                                      n_scales=args.n_scales_dis,
                                      sn=args.sn_dis)

        else:
            raise ValueError('Invalid dis_type!')

    @staticmethod
    def _create_gen(args):
        if args.gen_type == 'basic_res':
            return GeneratorBasicRes(ch=args.ch_gen,
                                     n_updownsample=args.n_updownsample_gen,
                                     n_res=args.n_res_gen,
                                     n_enhanced_upsample=args.n_enhanced_upsample_gen,
                                     n_mix_upsample=args.n_mix_upsample_gen,
                                     block_type=args.block_type_gen,
                                     upsample_type=args.upsample_type_gen)
        else:
            raise ValueError('Invalid gen_type!')

    @staticmethod
    def _create_gen_bw(args):
        if args.gen_type == 'basic_res':
            return GeneratorBasicRes(ch=args.ch_gen_bw,
                                     n_updownsample=args.n_updownsample_gen_bw,
                                     n_res=args.n_res_gen_bw,
                                     n_enhanced_upsample=args.n_enhanced_upsample_gen,
                                     n_mix_upsample=args.n_mix_upsample_gen,
                                     block_type=args.block_type_gen,
                                     upsample_type=args.upsample_type_gen)
        else:
            raise ValueError('Invalid gen_type!')

    def _fetch_data(self, dataset):
        gpu_device = '/gpu:0'
        imgdata = ImageData(self._img_size, self._augment_type)
        train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
        train_dataset = train_dataset.apply(shuffle_and_repeat(self._dataset_num)) \
            .apply(map_and_batch(imgdata.image_processing, self._batch_size,
                                 num_parallel_batches=16, drop_remainder=True)) \
            .apply(prefetch_to_device(gpu_device, None))
        train_iterator = train_dataset.make_one_shot_iterator()
        return train_iterator.get_next()

    def build_model_train(self):
        self._lr = tf.placeholder(tf.float32, name='learning_rate')

        # Input images
        self._domain_a = self._fetch_data(self._trainA_dataset)
        self._domain_b = self._fetch_data(self._trainB_dataset)

        # Forward generation
        self._x_ab = self._gen.translate(self._domain_a, scope='gen_a2b')

        # Backward generation
        if self._cyc_weight > 0.0:
            self._a_lowres = tf.image.resize_images(self._domain_a, [self._resolution_bw, self._resolution_bw])
            self._ab_lowres = tf.image.resize_images(self._x_ab, [self._resolution_bw, self._resolution_bw])
            self._aba_lowres = self._gen_bw.translate(self._ab_lowres, scope='gen_b2a')
        else:
            self._aba_lowres = tf.zeros([self._batch_size, self._resolution_bw, self._resolution_bw, 3])

            # Identity mapping
        self._x_bb = self._gen.translate(self._domain_b, reuse=True, scope='gen_a2b') \
            if self._id_weight > 0.0 else tf.zeros_like(self._domain_b)

        # Discriminator
        b_logits = self._dis.discriminate(self._domain_b, scope='dis_b')
        ab_logits = self._dis.discriminate(self._x_ab, reuse=True, scope='dis_b')

        # Adversarial loss for G
        adv_loss_gen_ab = self._adv_weight * adv_loss(ab_logits, self._gan_type, target='real')

        # Adversarial loss for D
        adv_loss_dis_b = self._adv_weight * adv_loss(b_logits, self._gan_type, target='real')
        adv_loss_dis_b += self._adv_weight * adv_loss(ab_logits, self._gan_type, target='fake')

        # Identity loss
        id_loss_bb = self._id_weight * l1_loss(self._domain_b, self._x_bb) \
            if self._id_weight > 0.0 else 0.0
        cyc_loss_aba = self._cyc_weight * l1_loss(self._a_lowres, self._aba_lowres) \
            if self._cyc_weight > 0.0 else 0.0

        # Weight decay
        reg_loss_gen = self._reg_weight * regularization_loss('gen_')
        reg_loss_dis = self._reg_weight * regularization_loss('dis_')

        # Overall loss
        self._gen_loss_all = adv_loss_gen_ab \
                             + id_loss_bb \
                             + cyc_loss_aba \
                             + reg_loss_gen

        self._dis_loss_all = adv_loss_dis_b \
                             + reg_loss_dis


        """ Training """
        t_vars = tf.trainable_variables()
        vars_gen = [var for var in t_vars if 'gen_' in var.name]
        vars_dis = [var for var in t_vars if 'dis_' in var.name]

        self._optim_gen = tf.train.AdamOptimizer(self._lr, beta1=0.5, beta2=0.999)\
            .minimize(self._gen_loss_all, var_list=vars_gen)
        self._optim_dis = tf.train.AdamOptimizer(self._lr, beta1=0.5, beta2=0.999)\
            .minimize(self._dis_loss_all, var_list=vars_dis)

        """" Summary """
        # Record the IN scaling factor for each residual block.
        summary_scale_res = summary_by_keywords(['gamma', 'resblock', 'res2'], node_type='variable')
        summary_logits_gen = summary_by_keywords('pre_tanh', node_type='tensor')
        summary_logits_dis = summary_by_keywords(['D_logits_'], node_type='tensor')

        summary_list_gen = []
        summary_list_gen.append(tf.summary.scalar("gen_loss_all", self._gen_loss_all))
        summary_list_gen.append(tf.summary.scalar("adv_loss_gen_ab", adv_loss_gen_ab))
        summary_list_gen.append(tf.summary.scalar("id_loss_bb", id_loss_bb))
        summary_list_gen.append(tf.summary.scalar("cyc_loss_aba", cyc_loss_aba))
        summary_list_gen.append(tf.summary.scalar("reg_loss_gen", reg_loss_gen))
        summary_list_gen.extend(summary_scale_res)
        summary_list_gen.extend(summary_logits_gen)
        self._summary_gen = tf.summary.merge(summary_list_gen)

        summary_list_dis = []
        summary_list_dis.append(tf.summary.scalar("dis_loss_all", self._dis_loss_all))
        summary_list_dis.append(tf.summary.scalar("adv_loss_dis_b", adv_loss_dis_b))
        summary_list_dis.append(tf.summary.scalar("reg_loss_dis", reg_loss_dis))
        summary_list_dis.extend(summary_logits_dis)
        self._summary_dis = tf.summary.merge(summary_list_dis)

    def build_model_test(self):
        self._test_domain_a = tf.placeholder(tf.float32, [1, self._img_size, self._img_size, 3],
                                            name='test_domain_A')
        test_fake_b = self._gen.translate(self._test_domain_a, scope='gen_a2b')
        self._test_fake_b = tf.identity(test_fake_b, 'test_fake_B')

    def train(self):
        tf.global_variables_initializer().run()
        self._saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self._log_dir, self._sess.graph)

        # restore the checkpoint if it exits
        could_load, checkpoint_counter = self._load(self._checkpoint_dir)
        if could_load:
            counter = checkpoint_counter + 1
            start_step = counter // self._n_iters_per_step
            start_batch_id = counter - start_step * self._n_iters_per_step
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            start_step = 0
            start_batch_id = 0
            print(" [!] Load failed...")

        # Looping over steps
        start_time = time.time()
        for step in range(start_step, self._n_steps):
            lr = self._init_lr if step < self._decay_step else \
                self._init_lr * (self._n_steps - step) / (self._n_steps - self._decay_step)
            for batch_id in range(start_batch_id, self._n_iters_per_step):
                train_feed_dict = {
                    self._lr: lr
                }

                # Update D
                loss_dis, summary_str_dis, _ = self._sess.run([self._dis_loss_all, self._summary_dis, self._optim_dis],
                                                              feed_dict=train_feed_dict)

                # Update G
                batch_a_images, batch_b_images, fake_b, identity_b, aba_lowres, loss_gen, summary_str_gen, _ = \
                    self._sess.run([self._domain_a, self._domain_b,
                                    self._x_ab, self._x_bb, self._aba_lowres,
                                    self._gen_loss_all, self._summary_gen, self._optim_gen],
                                   feed_dict=train_feed_dict)

                # display training status
                print("Step: [%2d] [%5d/%5d] time: %4.4f D_loss: %.8f, G_loss: %.8f"
                      % (step, batch_id, self._n_iters_per_step, time.time() - start_time, loss_dis, loss_gen))

                if (counter+1) % self._summary_freq == 0:
                    writer.add_summary(summary_str_dis, counter)
                    writer.add_summary(summary_str_gen, counter)

                if (counter+1) % self._img_save_freq == 0:
                    aba_lowres_resize = batch_resize(aba_lowres, self._img_size)
                    merged = np.vstack([batch_a_images, fake_b, aba_lowres_resize, batch_b_images, identity_b])
                    save_images(merged, [5, self._batch_size],
                                os.path.join(self._sample_dir, 'sample_{:03d}_{:05d}.jpg'.format(step, batch_id)))

                if (counter+1) % self._ckpt_save_freq == 0:
                    self._save(self._checkpoint_dir, counter)

                counter += 1

            # After each step, start_batch_id is set to zero.
            # Non-zero value is only for the first step after loading a pre-trained model.
            start_batch_id = 0

            # Save the final model.
            self._save(self._checkpoint_dir, counter-1)

    def test(self):
        tes_a_dir = os.path.join(os.path.dirname(__file__), 'dataset', self._test_dataset_name, 'testA')
        test_a_files = get_img_paths(tes_a_dir, self._dataset_struct)

        if self._saver is None:
            self._saver = tf.train.Saver()
        could_load, checkpoint_counter = self._load(self._checkpoint_dir)
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
        could_load, checkpoint_counter = self._load(self._checkpoint_dir)

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

    def _save(self, checkpoint_dir, step):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._saver.save(self._sess, os.path.join(checkpoint_dir, self._model_name + '.model'), global_step=step)

    def _load(self, checkpoint_dir):
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
