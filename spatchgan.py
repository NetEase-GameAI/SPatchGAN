import os
from utils import get_img_paths
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
        self.img_ch = args.img_ch
        self.augment_flag = args.augment_flag
        trainA_dir = os.path.join(os.path.dirname(__file__), 'dataset', self.dataset_name, 'trainA')
        trainB_dir = os.path.join(os.path.dirname(__file__), 'dataset', self.dataset_name, 'trainB')
        self.trainA_dataset = get_img_paths(trainA_dir)
        self.trainB_dataset = get_img_paths(trainB_dir)
        # Auto detect the 2nd level if there is no image at the 1st level.
        if len(self.trainA_dataset) == 0 or len(self.trainB_dataset) == 0:
            self.trainA_dataset = get_img_paths(trainA_dir, level=2)
            self.trainB_dataset = get_img_paths(trainB_dir, level=2)

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
                                           n_downsample_init=args.n_downsample_init,
                                           n_scales=args.n_scales,
                                           n_adapt=args.n_adapt,
                                           n_mix=args.n_mix,
                                           logits_type=args.logits_type_dis,
                                           stats=args.stats,
                                           sn=args.sn)
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
