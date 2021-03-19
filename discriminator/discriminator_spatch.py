import tensorflow as tf


class DiscriminatorSPatch:
    def __init__(self, ch, n_downsample_init, n_scales, n_adapt, n_mix,
                 logits_type: str, stats: list, sn):
        self._ch = ch
        self.n_downsample_init = n_downsample_init
        self._n_scales = n_scales
        self._n_adapt = n_adapt
        self.n_mix = n_mix
        self._logits_type = logits_type
        self._stats = stats
        self.sn = sn

    def discriminate(self, x):
        return 1.0