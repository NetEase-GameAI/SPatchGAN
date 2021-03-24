import tensorflow as tf
from ops import conv, lrelu, global_avg_pooling, global_max_pooling, fully_connected


class DiscriminatorSPatch:
    def __init__(self, ch, n_downsample_init, n_scales, n_adapt, n_mix,
                 logits_type: str, stats: list, sn):
        self._ch = ch
        self._n_downsample_init = n_downsample_init
        self._n_scales = n_scales
        self._n_adapt = n_adapt
        self._n_mix = n_mix
        self._logits_type = logits_type
        self._stats = stats
        self._sn = sn

    def discriminate(self, x, reuse=False, scope='dis'):
        with tf.variable_scope(scope, reuse=reuse):
            channel = self._ch
            logits_list = []

            for i in range(self._n_downsample_init):
                with tf.variable_scope('down_{}'.format(i)):
                    # (256, 256, 3) -> (128, 128, 256) -> (64, 64, 512)
                    x = conv(x, channel, kernel=4, stride=2, pad=1, sn=self._sn)
                    x = lrelu(x)
                    channel *= 2

            for i in range(self._n_scales):
                with tf.variable_scope('scale_{}'.format(i)):
                    # (64, 64, 512) -> (32, 32, 1024) -> (16, 16, 1024) -> (8, 8, 1024) -> (4, 4, 1024)
                    x = conv(x, channel, kernel=4, stride=2, pad=1, sn=self._sn, scope='conv_k4')
                    x = lrelu(x)
                    logits = self._dis_logits(x)
                    logits_list.extend(logits)

            return logits_list

    def _dis_logits(self, x, scope='dis_logits'):
        if self._logits_type == 'stats':
            return self._dis_logits_stats(x, scope=scope)
        else:
            raise ValueError('Invalid logits_type_dis!')

    def _dis_logits_stats(self, x, scope='dis_logits'):
        with tf.variable_scope(scope):
            logits_list = []
            channel = x.shape[-1].value

            for i in range(self._n_adapt):
                with tf.variable_scope('premix_{}'.format(i)):
                    x = conv(x, channel, sn=self._sn)
                    x = lrelu(x)

            if 'mean' in self._stats:
                with tf.variable_scope('gap'):
                    x_gap = global_avg_pooling(x)
                    x_gap_logits = self._mlp_logits(x_gap)
                    x_gap_logits = tf.identity(x_gap_logits, 'D_logits_gap')
                    logits_list.append(x_gap_logits)

            if 'max' in self._stats:
                with tf.variable_scope('gmp'):
                    x_gmp = global_max_pooling(x)
                    x_gmp_logits = self._mlp_logits(x_gmp)
                    x_gmp_logits = tf.identity(x_gmp_logits, 'D_logits_gmp')
                    logits_list.append(x_gmp_logits)

            if 'stddev' in self._stats:
                with tf.variable_scope('stddev'):
                    # Calculate the channel-wise uncorrected standard deviation
                    x_diff_square = tf.square(x - tf.reduce_mean(x, axis=[1, 2], keepdims=True))
                    x_stddev = tf.sqrt(global_avg_pooling(x_diff_square))
                    x_stddev_logits = self._mlp_logits(x_stddev)
                    x_stddev_logits = tf.identity(x_stddev_logits, 'D_logits_stddev')
                    logits_list.append(x_stddev_logits)

            return logits_list

    def _mlp_logits(self, x, n_ch=None, scope='dis_logits_mix'):
        with tf.variable_scope(scope):
            shape = x.shape.as_list()
            channel = n_ch or shape[-1]
            if len(shape) == 2:
                for i in range(self._n_mix):
                    x = fully_connected(x, units=channel, sn=self._sn, scope='mix_'+str(i))
                    x = lrelu(x)
                x = fully_connected(x, units=1, sn=self._sn, scope='logits')
            elif len(shape) == 4:
                for i in range(self._n_mix):
                    x = conv(x, channels=channel, kernel=1, stride=1, sn=self._sn, scope='mix_'+str(i))
                    x = lrelu(x)
                x = conv(x, channels=1, kernel=1, stride=1, sn=self._sn, scope='logits')
            return x

