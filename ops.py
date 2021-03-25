import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)


def conv(x, channels, kernel=1, stride=1, pad=0, pad_type: str = 'zero', use_bias=True,
         sn: str = None, scope: str = 'conv_0'):
    """Convolution layer."""
    with tf.variable_scope(scope):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                # For kernel = 3, stride = 2, pad=1:
                # pad_top = pad_left = 1
                # pad_bottom = pad_right = 0
                # h_out = (h_in + 1 - 3) / 2 + 1 = h_in / 2
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn is not None:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w, method=sn),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)
        return x


def fully_connected(x, units, use_bias=True, sn: str = None, scope='linear'):
    """Fully connected layer."""
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn is not None:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w, method=sn)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w, method=sn))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


def instance_norm(x, scope='instance_norm'):
    """Instance normalization layer."""
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm'):
    """Layer normalization layer."""
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def spectral_norm(w, n_iters=1, method: str = 'fast'):
    """Spectral normalization layer."""
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = tf.stop_gradient(u) if method == 'full' else u
    v_hat = None
    for i in range(n_iters):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    if method == 'fast':
        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)
    elif method == 'full':
        pass
    else:
        raise RuntimeError('Invalid sn method!')

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def lrelu(x, alpha=0.2):
    """Leaky ReLU."""
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    """ReLU."""
    return tf.nn.relu(x)


def tanh(x):
    """Tanh."""
    return tf.tanh(x)


def global_avg_pooling(x):
    """Global average pooling for the NHWC data."""
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap


def global_max_pooling(x):
    """Global max pooling for the NHWC data."""
    gmp = tf.reduce_max(x, axis=[1, 2])
    return gmp


def nearest_up(x, scale_factor=2):
    """Nearest neighbor upsampling."""
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def bilinear_up(x, scale_factor=2):
    """Bilinear upsampling."""
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_images(x, size=new_size)


def resblock_v1(x_init, channel, pad_type: str = 'zero', use_bias=True, is_res=True, scope='resblock_0'):
    """Residual block."""
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channel, kernel=3, pad=1, pad_type=pad_type, use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channel, kernel=3, pad=1, pad_type=pad_type, use_bias=use_bias)
            x = instance_norm(x)

        x_ret = x + x_init if is_res else x
        return x_ret


def l1_loss(x, y):
    """Calculate the L1 loss."""
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def regularization_loss(scope_name: str):
    """Collect the regularization loss."""
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization :
        if scope_name in item.name :
            loss.append(item)

    return tf.reduce_sum(loss)


def adv_loss(x, loss_func : str, target : str):
    """Calculate the adversarial loss."""
    loss_list = []
    logits_list = x if isinstance(x, list) else [x]
    for i, logits in enumerate(logits_list):
        if loss_func == 'lsgan':
            if target == 'real':
                target_val = 1.0
            elif target == 'fake':
                target_val = 0.0
            else:
                raise ValueError('Invalid target {} for adv_loss'.format(target))
            loss = tf.squared_difference(logits, target_val)
        else:
            raise ValueError('Invalid loss_func {} for adv_loss'.format(loss_func))
        loss = tf.reduce_mean(loss) / len(logits_list)
        loss_list.append(loss)

    return sum(loss_list)
