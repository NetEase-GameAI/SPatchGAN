import argparse
import os
import tensorflow as tf
# from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Tensorflow implementation of SPatchGAN."
    parser = argparse.ArgumentParser(description=desc)

    # General configs
    parser.add_argument('--network', type=str, default='spatchgan', help='Network type: [spatchgan].')
    parser.add_argument('--phase', type=str, default='train',
                        help='Phase: [train / test / freeze_graph].')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the training dataset.')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='Name of the testing dataset. Same as the training dataset by default.')
    parser.add_argument('--dataset_type', type=str, default='plain', help='Dataset type: [plain / tree].')
    parser.add_argument('--suffix', type=str, default=None, help='suffix for the model name.')

    # Training configs
    parser.add_argument('--n_steps', type=int, default=50, help='Number of training steps.')
    parser.add_argument('--n_iters_per_step', type=int, default=10000, help='Number of iterations per step')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--img_save_freq', type=int, default=1000, help='Image saving frequency in iteration.')
    parser.add_argument('--ckpt_save_freq', type=int, default=1000, help='Checkpoint saving frequency in iteration.')
    parser.add_argument('--summary_freq', type=int, default=100, help='Tensorflow summary frequency.')
    parser.add_argument('--decay_step', type=int, default=10, help='Starting point for learning rate decay.')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate.')
    parser.add_argument('--adv_weight', type=float, default=4.0, help='Adversarial loss weight.')
    parser.add_argument('--reg_weight', type=float, default=1.0, help='Regularization weight.')
    parser.add_argument('--cyc_weight', type=float, default=20.0, help='Weak cycle loss weight.')
    parser.add_argument('--id_weight', type=int, default=10, help='Identity loss weight.')
    parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type: [lsgan].')

    # Input configs
    parser.add_argument('--img_size', type=int, default=256, help='The size of input images.')
    parser.add_argument('--img_ch', type=int, default=3, help='Number of channels of the input images.')
    parser.add_argument('--augment_flag', type=str, default='pad_crop',
                        help='Augmentation method: [pad_crop / resize_crop / disabled].')

    # Discriminator configs
    parser.add_argument('--dis_type', type=str, default='spatch', help='D type: [spatch / patch].')
    parser.add_argument('--logits_type_dis', type=str, default='stats', help='D logits calculation method: [stats].')
    parser.add_argument('--ch_dis', type=int, default=256, help='Base channel number of D.')
    parser.add_argument('--n_downsample_init_dis', type=int, default=6,
                        help='Number of downsampling layers in the initial feature extraction block.')
    parser.add_argument('--n_scales_dis', type=int, default=4, help='Number of scales in D.')
    parser.add_argument('--sn_type_dis', type=str, default='lean', help='Spectral norm type: [lean / full / disabled]')
    parser.add_argument('--n_adapt_dis', type=int, default=2, help='Number of layers in each adaptation block.')
    parser.add_argument('--n_mix_dis', type=int, default=2, help='Number of mixing layers in each MLP.')
    parser.add_argument('--pad_type_dis', type=str, default='zero', help='Padding type in D: [zero / reflect]')
    parser.add_argument('--gap_dis', type=str2bool, default=True, help='Use the gap output in D.')
    parser.add_argument('--gmp_dis', type=str2bool, default=True, help='Use the gmp output in D.')
    parser.add_argument('--stddev_dis', type=str2bool, default=True, help='Use the stddev output in D.')

    # Generator configs
    parser.add_argument('--gen_type', type=str, default='basic_res', help='G type: [basic_res].')
    parser.add_argument('--block_type_gen', type=str, default='v1', help='G residual block type: [v1].')
    parser.add_argument('--ch_gen', type=int, default=128, help='Base channel number of forward G.')
    parser.add_argument('--ch_gen_bw', type=int, default=512, help='Base channel number of backward G.')
    parser.add_argument('--upsample_type_gen', type=str, default='nearest', help='Upsampling method: [nearest].')
    parser.add_argument('--n_updownsample_gen', type=int, default=3,
                        help='Number of up/downsampling layers in forward G.')
    parser.add_argument('--n_updownsample_gen_bw', type=int, default=3,
                        help='Number of up/downsampling layers in backward G.')
    parser.add_argument('--n_res_gen', type=int, default=8, help='Number of residual blocks in forward G.')
    parser.add_argument('--n_res_gen_bw', type=int, default=8, help='Number of residual blocks in backward G.')
    parser.add_argument('--n_enhanced_upsample_blocks_gen', type=int, default=1,
                        help='Number of enhanced upsampling blocks that include multiple mixing layers.')
    parser.add_argument('--n_mix_upsample_gen', type=int, default=2,
                        help='Number of mixing layers in an enhanced upsampling block.')
    parser.add_argument('--init_kernel_size_gen', type=int, default=3, help='Init kernel size in G')
    parser.add_argument('--resize_factor_gen_bw', type=int, default=8,
                        help='The resizing factor of input images for backward G.')

    # Directory names
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output'),
                        help='Directory name to save all output')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint.',
                        help='Directory to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='gen',
                        help='Directory to save the generated images.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save training logs.')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory to save the samples on training.')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args is None:
        exit()

    # open session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:

        if args.network == 'spatchgan':
            from SPATCHGAN import SPATCHGAN
            gan = SPATCHGAN(sess, args)
        else:
            raise RuntimeError('Invalid network!')

        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")
        elif args.phase == 'test':
            gan.test()
            print(" [*] Test finished!")
        elif args.phase == 'freeze_graph':
            gan.freeze_graph()
            print(" [*] Graph frozen!")
        else:
            raise RuntimeError('Invalid phase!')


if __name__ == '__main__':
    main()
