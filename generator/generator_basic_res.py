import tensorflow as tf
from ops import conv, instance_norm, layer_norm, relu, tanh, resblock_v1, nearest_up, bilinear_up


class GeneratorBasicRes:
    def __init__(self, ch, n_updownsample, n_res, n_enhanced_upsample, n_mix_upsample, block_type, upsample_type):
        self._ch = ch
        self._n_updownsample = n_updownsample
        self._n_res = n_res
        self._n_enhanced_upsample = n_enhanced_upsample
        self._n_mix_upsample = n_mix_upsample
        self._block_type = block_type
        self._upsample_type = upsample_type

    def translate(self, x, reuse=False, scope='gen'):
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self._ch

            # Down-Sampling
            for i in range(self._n_updownsample):
                with tf.variable_scope('down_{}'.format(i)):
                    # (256, 256, 3) -> (128, 128, 128) -> (64, 64, 256) -> (32, 32, 512)
                    x = conv(x, channel, kernel=3, stride=2, pad=1)
                    x = instance_norm(x)
                    if i < self._n_updownsample - 1:
                        x = relu(x)
                        channel *= 2

            if self._n_updownsample == 0:
                with tf.variable_scope('mix_init'):
                    x = conv(x, channel, kernel=3, pad=1)
                    x = instance_norm(x)

            for i in range(self._n_res):
                with tf.variable_scope('res_{}'.format(i)):
                    x = self._conv_block(x, block_type=self._block_type, channel=channel)

            for i in range(self._n_updownsample):
                with tf.variable_scope('up_{}'.format(i)):
                    # (32, 32, 512) -> (64, 64, 512) -> (128, 128, 256) -> (256, 256, 128)
                    x = self._upsample(x, method=self._upsample_type)
                    channel = channel if i < self._n_enhanced_upsample else channel // 2
                    n_mix_upsample = self._n_mix_upsample if i < self._n_enhanced_upsample else 1
                    for j in range(n_mix_upsample):
                        with tf.variable_scope('mix_{}'.format(j)):
                            x = conv(x, channel, kernel=3, stride=1, pad=1)
                            x = layer_norm(x)
                            x = relu(x)

            if self._n_updownsample == 0:
                with tf.variable_scope('mix_end'):
                    x = conv(x, channel, kernel=3, stride=1, pad=1)
                    x = layer_norm(x)
                    x = relu(x)

            with tf.variable_scope('logits'):
                # (256, 256, 128) -> (256, 256, 3)
                x = conv(x, channels=3, kernel=3, pad=1, scope='G_logit')
                x = tanh(x)

            return x

    @classmethod
    def _conv_block(cls, x, block_type, channel, scope='resblock_0'):
        if block_type == 'v1':
            x = resblock_v1(x, channel=channel, scope=scope)
        else:
            raise ValueError('Wrong block_type!')
        return x

    @classmethod
    def _upsample(cls, x, method: str = 'nearest'):
        if method == 'nearest':
            x = nearest_up(x)
        elif method == 'bilinear':
            x = bilinear_up(x)
        else:
            raise ValueError('Invalid upsampling method!')
        return x
