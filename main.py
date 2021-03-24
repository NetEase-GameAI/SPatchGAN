import os
import tensorflow as tf
from configs import parse_args
from utils import show_all_variables
from spatchgan import SPatchGAN


def main():
    # parse arguments
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:

        if args.network == 'spatchgan':
            gan = SPatchGAN(sess, args)
        else:
            raise RuntimeError('Invalid network!')

        if args.phase == 'train':
            gan.build_model_train()
            show_all_variables()
            gan.train()
            print(" [*] Training finished!")
        elif args.phase == 'test':
            gan.build_model_test()
            show_all_variables()
            gan.test()
            print(" [*] Test finished!")
        elif args.phase == 'freeze_graph':
            gan.build_model_test()
            show_all_variables()
            gan.freeze_graph()
            print(" [*] Graph frozen!")
        else:
            raise RuntimeError('Invalid phase!')


if __name__ == '__main__':
    main()
