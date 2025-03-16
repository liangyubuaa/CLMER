class Config(object):
    def __init__(self):
        self.input_channels = 17
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 4
        self.dropout = 0.35
        self.features_len = 18

        self.num_epoch = 1000

        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        self.drop_last = False
        self.batch_size = 80

        self.Context_Cont = Context_Cont_configs()
        self.timeseries = timeseries()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class timeseries(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 2
