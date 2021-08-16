import os
import sys
import cv2
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import get_img_paths
from frozen_model.image_translator import ImageTranslator

cv2.setNumThreads(1)


def _get_images(img_loc: str) -> list:
    """ Find the image(s) in a given location.
    :param img_loc: either an image file path, or a directory containing images.
    :return: a list of image paths.
    """
    image_list = []
    if os.path.isfile(img_loc):
        image_list.append(img_loc)
    elif os.path.isdir(img_loc):
        image_list.extend(get_img_paths(img_loc))
    return image_list


def main(args):
    """ Translate all images in a given location with a frozen model (.pb)."""
    images = _get_images(args.image)
    if len(images) == 0:
        raise RuntimeError('No image in {}!'.format(args.image))

    size = 256
    translator = ImageTranslator(model_path=args.model,
                                 size=size,
                                 n_threads_intra=args.n_threads_intra,
                                 n_threads_inter=args.n_threads_inter)
    os.makedirs(args.output_dir, exist_ok=True)

    st = time.time()
    for i, image in enumerate(images):
        print('{}: {}'.format(i, image))
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            print('Invalid image {}'.format(image))
            continue
        else:
            output = None
            for i_iter in range(args.n_iters):
                print('Iter: {}'.format(i_iter))
                output = translator.translate(img)
            save_path = os.path.join(args.output_dir, os.path.basename(image))
            cv2.imwrite(save_path, output)

    time_total = time.time() - st
    time_cost_per_img_ms = int((time_total / len(images) / args.n_iters) * 1000)
    print('Time cost per image: {} ms'.format(time_cost_per_img_ms))


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Image file path or directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--model', type=str, required=True, help='.pb model path.')
    parser.add_argument('--n_threads_inter', type=int, default=1, help='Number of inter op threads.')
    parser.add_argument('--n_threads_intra', type=int, default=1, help='Number of intra op threads.')
    parser.add_argument('--n_iters', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    main(_parse_arguments())