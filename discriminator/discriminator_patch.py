import tensorflow as tf
from ops import conv, lrelu


# Multiscale patchgan discriminator for MUNIT / Council-GAN / ACL-GAN
class DiscriminatorPatch:
    def __init__(self, ch, n_downsample_init, n_scales, sn):
        self._ch = ch  # 64 in MUNIT
        self._n_downsample_init = n_downsample_init  # 4 in MUNIT
        self._n_scales = n_scales  # 3 in MUNIT
        self._sn = sn

    def discriminate(self, x, reuse=False, scope='dis'):
        with tf.variable_scope(scope, reuse=reuse):
            logits = []
            for i in range(self._n_scales):
                logits_patch = self._discriminator_per_scale(x, reuse=reuse, scope='scale_{}'.format(i))
                logits.append(logits_patch)
                x = tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')
            return logits

    def _discriminator_per_scale(self, x, reuse=False, scope='scale_0'):
        with tf.variable_scope(scope, reuse=reuse):
            channel = self._ch
            for i in range(self._n_downsample_init):
                with tf.variable_scope('down_{}'.format(i)):
                    x = conv(x, channel, kernel=4, stride=2, pad=1, sn=self._sn)
                    x = lrelu(x, 0.2)
                    channel *= 2
            x = conv(x, channels=1, kernel=1, stride=1, sn=self._sn, scope='logits')
            x = tf.identity(x, 'D_logits_patch')
            return x
