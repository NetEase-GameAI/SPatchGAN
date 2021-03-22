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
from generator.generator_basic_res import GeneratorBasicRes


class SPatchGAN:
    def __init__(self, sess, args):
        # General
        self.model_name = 'SPatchGAN'
        self.sess = sess
        self.saver = None
        self.phase = args.phase
        self.dataset_name = args.dataset
        self.test_dataset_name = args.test_dataset or args.dataset
        self.dataset_struct = args.dataset_struct
        self.suffix = args.suffix

        # Training
        self.n_steps = args.n_steps
        self.n_iters_per_step = args.n_iters_per_step
        self.batch_size = args.batch_size
        self.img_save_freq = args.img_save_freq
        self.ckpt_save_freq = args.ckpt_save_freq
        self.summary_freq = args.summary_freq
        self.decay_step = args.decay_step
        self.init_lr = args.lr
        self.adv_weight = args.adv_weight
        self.reg_weight = args.reg_weight
        self.cyc_weight = args.cyc_weight
        self.id_weight = args.id_weight
        self.gan_type = args.gan_type

        # Input
        self.img_size = args.img_size
        self.augment_flag = args.augment_flag
        trainA_dir = os.path.join(os.path.dirname(__file__), 'dataset', self.dataset_name, 'trainA')
        trainB_dir = os.path.join(os.path.dirname(__file__), 'dataset', self.dataset_name, 'trainB')
        self.trainA_dataset = get_img_paths(trainA_dir, self.dataset_struct)
        self.trainB_dataset = get_img_paths(trainB_dir, self.dataset_struct)
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        # Discriminator
        if args.dis_type == 'spatch':
            stats = []
            if args.mean_dis:
                stats.append('mean')
            if args.max_dis:
                stats.append('max')
            if args.mean_dis:
                stats.append('stddev')
            self.dis = DiscriminatorSPatch(ch=args.ch_dis,
                                           n_downsample_init=args.n_downsample_init_dis,
                                           n_scales=args.n_scales_dis,
                                           n_adapt=args.n_adapt_dis,
                                           n_mix=args.n_mix_dis,
                                           logits_type=args.logits_type_dis,
                                           stats=stats,
                                           sn=args.sn_dis)
        else:
            raise ValueError('Invalid dis_type!')

        # Generator
        if args.gen_type == 'basic_res':
            self.gen = GeneratorBasicRes(ch=args.ch_gen,
                                         n_updownsample=args.n_updownsample_gen,
                                         n_res=args.n_res_gen,
                                         n_enhanced_upsample=args.n_enhanced_upsample_gen,
                                         n_mix_upsample=args.n_mix_upsample_gen,
                                         block_type=args.block_type_gen,
                                         upsample_type=args.upsample_type_gen)
            self.gen_bw = GeneratorBasicRes(ch=args.ch_gen_bw,
                                            n_updownsample=args.n_updownsample_gen_bw,
                                            n_res=args.n_res_gen_bw,
                                            n_enhanced_upsample=args.n_enhanced_upsample_gen,
                                            n_mix_upsample=args.n_mix_upsample_gen,
                                            block_type=args.block_type_gen,
                                            upsample_type=args.upsample_type_gen)
        else:
            raise ValueError('Invalid gen_type!')
        self.resolution_bw = self.img_size // args.resize_factor_gen_bw

        # Directory
        self.output_dir = args.output_dir
        self.model_dir = "{}_{}_{}".format(self.model_name, self.dataset_name, self.suffix)
        self.checkpoint_dir = os.path.join(self.output_dir, self.model_dir, args.checkpoint_dir)
        self.sample_dir = os.path.join(self.output_dir, self.model_dir, args.sample_dir)
        self.log_dir = os.path.join(self.output_dir, self.model_dir, args.log_dir)
        self.result_dir = os.path.join(self.output_dir, self.model_dir, args.result_dir)
        for dir in [self.checkpoint_dir, self.sample_dir, self.log_dir, self.result_dir]:
            os.makedirs(dir, exist_ok=True)

        print()
        print('##### Information #####')
        print('Number of trainA/B images: {}/{}'.format(len(self.trainA_dataset), len(self.trainB_dataset)) )
        print()

    def fetch_data(self, dataset):
        gpu_device = '/gpu:0'
        Image_Data_Class = ImageData(self.img_size, self.augment_flag)
        train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
        train_dataset = train_dataset.apply(shuffle_and_repeat(self.dataset_num)) \
            .apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size,
                                 num_parallel_batches=16, drop_remainder=True)) \
            .apply(prefetch_to_device(gpu_device, None))
        train_iterator = train_dataset.make_one_shot_iterator()
        return train_iterator.get_next()

    def build_model_train(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # Input images
        self.domain_A = self.fetch_data(self.trainA_dataset)
        self.domain_B = self.fetch_data(self.trainB_dataset)

        # Forward generation
        self.x_ab = self.gen.translate(self.domain_A, scope='gen_a2b')

        # Backward generation
        if self.cyc_weight > 0.0:
            self.a_lr = tf.image.resize_images(self.domain_A, [self.resolution_bw, self.resolution_bw])
            self.ab_lr = tf.image.resize_images(self.x_ab, [self.resolution_bw, self.resolution_bw])
            self.aba_lr = self.gen_bw.translate(self.ab_lr, scope='gen_b2a')
        else:
            self.aba_lr = tf.zeros([self.batch_size, self.resolution_bw, self.resolution_bw, 3])

            # Identity mapping
        self.x_bb = self.gen.translate(self.domain_B, reuse=True, scope='gen_a2b') \
            if self.id_weight > 0.0 else tf.zeros_like(self.domain_B)

        # Discriminator
        b_logits = self.dis.discriminate(self.domain_B, scope='dis_b')
        ab_logits = self.dis.discriminate(self.x_ab, reuse=True, scope='dis_b')

        # Adversarial loss for G
        self.adv_loss_gen_ab = self.adv_weight * adv_loss(ab_logits, self.gan_type, target='real')

        # Adversarial loss for D
        self.adv_loss_dis_b = self.adv_weight * adv_loss(b_logits, self.gan_type, target='real')
        self.adv_loss_dis_b += self.adv_weight * adv_loss(ab_logits, self.gan_type, target='fake')

        # Identity loss
        self.id_loss_bb = self.id_weight * l1_loss(self.domain_B, self.x_bb) \
            if self.id_weight > 0.0 else 0.0
        self.cyc_loss_aba = self.cyc_weight * l1_loss(self.a_lr, self.aba_lr) \
            if self.cyc_weight > 0.0 else 0.0

        # Weight decay
        self.reg_loss_gen = self.reg_weight * regularization_loss('gen_')
        self.reg_loss_dis = self.reg_weight * regularization_loss('dis_')

        # Overall loss
        self.gen_loss_all = self.adv_loss_gen_ab + \
                            self.id_loss_bb + \
                            self.cyc_loss_aba + \
                            self.reg_loss_gen

        self.dis_loss_all = self.adv_loss_dis_b + \
                            self.reg_loss_dis


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'gen_' in var.name]
        D_vars = [var for var in t_vars if 'dis_' in var.name]

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999)\
            .minimize(self.gen_loss_all, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999)\
            .minimize(self.dis_loss_all, var_list=D_vars)

        """" Summary """
        # Record the IN scaling factor for each residual block.
        summary_scale_res = summary_by_keywords(['gamma', 'resblock', 'res2'], node_type='variable')
        summary_logits_gen = summary_by_keywords('pre_tanh', node_type='tensor')
        summary_logits_dis = summary_by_keywords(['D_logits_'], node_type='tensor')

        summary_list_gen = []
        summary_list_gen.append(tf.summary.scalar("gen_loss_all", self.gen_loss_all))
        summary_list_gen.append(tf.summary.scalar("adv_loss_gen_ab", self.adv_loss_gen_ab))
        summary_list_gen.append(tf.summary.scalar("reg_loss_gen", self.reg_loss_gen))
        summary_list_gen.extend(summary_scale_res)
        summary_list_gen.extend(summary_logits_gen)
        self.summary_gen = tf.summary.merge(summary_list_gen)

        summary_list_dis = []
        summary_list_dis.append(tf.summary.scalar("dis_loss_all", self.dis_loss_all))
        summary_list_dis.append(tf.summary.scalar("adv_loss_dis_b", self.adv_loss_dis_b))
        summary_list_dis.append(tf.summary.scalar("reg_loss_dis", self.reg_loss_dis))
        summary_list_dis.extend(summary_logits_dis)
        self.summary_dis = tf.summary.merge(summary_list_dis)

    def build_model_test(self):
        self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, 3],
                                            name='test_domain_A')
        test_fake_B = self.gen.translate(self.test_domain_A, scope='gen_a2b')
        self.test_fake_B = tf.identity(test_fake_B, 'test_fake_B')

    def train(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_step = checkpoint_counter // self.n_steps
            start_batch_id = checkpoint_counter - start_step * self.n_iters_per_step
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_step = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # Looping over steps
        start_time = time.time()
        for step in range(start_step, self.n_steps):
            lr = self.init_lr if step < self.decay_step else \
                self.init_lr * (self.n_steps - step) / (self.n_steps - self.decay_step)
            for idx in range(start_batch_id, self.n_iters_per_step):
                train_feed_dict = {
                    self.lr: lr
                }

                # Update D
                d_loss, summary_str, _ = self.sess.run([self.dis_loss_all, self.summary_dis, self.D_optim],
                                                       feed_dict=train_feed_dict)
                if (idx+1) % self.summary_freq == 0:
                    self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_images, batch_B_images, fake_B, identity_B, ABA_lr, g_loss, summary_str, _ = \
                    self.sess.run([self.domain_A, self.domain_B,
                                   self.x_ab, self.x_bb, self.aba_lr,
                                   self.gen_loss_all, self.summary_gen, self.G_optim],
                                  feed_dict=train_feed_dict)

                if (idx+1) % self.summary_freq == 0:
                    self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Step: [%2d] [%5d/%5d] time: %4.4f D_loss: %.8f, G_loss: %.8f"
                      % (step, idx, self.n_iters_per_step, time.time() - start_time, d_loss, g_loss))

                if (idx+1) % self.img_save_freq == 0:
                    ABA_lr_resize = batch_resize(ABA_lr)
                    merged = np.vstack([batch_A_images, fake_B, ABA_lr_resize, batch_B_images, identity_B])
                    save_images(merged, [5, self.batch_size],
                                os.path.join(self.sample_dir, 'sample_{:03d}_{:05d}.jpg'.format(step, idx + 1)))

                if (idx+1) % self.ckpt_save_freq == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    def test(self):
        testA_dir = os.path.join(os.path.dirname(__file__), 'dataset', self.test_dataset_name, 'testA')
        test_A_files = get_img_paths(testA_dir, self.dataset_struct)

        if self.saver is None:
            self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")
            raise RuntimeError("Failed to load the checkpoint")

        dataset_tag = '' if self.test_dataset_name == self.dataset_name else self.test_dataset_name + '_'
        result_dir = os.path.join(self.result_dir, dataset_tag + str(checkpoint_counter))
        os.makedirs(result_dir, exist_ok=True)

        st = time.time()
        for sample_file in test_A_files:  # A -> B
            print('Processing source image: ' + sample_file)
            input = load_test_data(sample_file, size=self.img_size)
            fake_img = self.sess.run(self.test_fake_B, feed_dict={self.test_domain_A: input})

            if self.dataset_struct == 'plain':
                dst_dir = result_dir
            elif self.dataset_struct == 'tree':
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
        time_cost_per_img_ms = round(time_cost * 1000 / len(test_A_files))
        print('Time cost per image: {} ms'.format(time_cost_per_img_ms))

    def freeze_graph(self):
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError("Failed to load the checkpoint")

        output_dir = os.path.join(self.checkpoint_dir, 'pb')
        os.makedirs(output_dir, exist_ok=True)
        time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_file = os.path.join(output_dir, 'output_graph_' + time_stamp + '.pb')

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=self.sess,
            input_graph_def=self.sess.graph_def,
            output_node_names=['test_fake_B'])

        # Save the frozen graph
        with open(output_file, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    def save(self, checkpoint_dir, step):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
