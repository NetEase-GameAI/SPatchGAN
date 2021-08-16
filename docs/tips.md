## Hyperparameter Tuning

- Discriminator
    - We set the number of scales (``n_scales_dis``) to be a constant (4) in the paper. This is to verify that a good performance for different tasks can be achieved with a fixed network structure. We found in practice that a reduced number of scales, e.g., 3, is often discriminative enough and more stabilized, especially for the tasks which require a significant shape deformation. In such case, the 4-th scale, which is the most discriminative one in the default setting, tends to be an overkill.
    - Reducing the number of base channels (``ch_dis``) is an effective way to accelerate the training process.

- Generator
    - To improve the inference speed, you may want to reduce the number of base channels (``ch_gen``) or the number of enhanced upsampling layers (``n_enhanced_upsample_gen``).

- Weak cycle
    - The weight for the cycle constraint (``cyc_weight``) can be adjusted on a per-task basis. A large value is generally more prohibitive for the shape deformation. On the other hand, it helps to keep the generated image correlated with the source image.
    - The input of the backward generator is resized by 1/(``resize_factor_gen_bw``). The number of downsampling and upsampling layers of the backward generator is set by ``n_updownsample_gen_bw``. These two parameters can be adjusted to change the weak cycle to a full resolution forward cycle, or something in between. The effect is somewhat similar to increasing the cycle weight.

- Training with a higher resolution
    - You may want to adjust several parameters if the input / output resolution is higher than 256x256. Take 512x512 as an example. A good starting point will be setting ``img_size`` to 512, increasing ``n_downsample_init_dis`` from 2 to 3, and reducing ``ch_dis``, ``ch_gen`` and ``ch_gen_bw`` by a factor of two.