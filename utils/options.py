class Options:
    def __init__(
        self,
        image_size=224,
        num_classes=2,
        use_pretrained=True,
        mlp_dr=0.3,
        lr=0.001,
        tune_last_layer_only=True,
        schedule_milestone=[],
        gpus=0,
        optimizer="adam",
        gfn_dropout=True,
        momentum=0.9,
        lr_decay=0.001,
        mask="none",
        sample_num=20,
        t_test=True,
        test_sample_mode="sample",
        use_t_in_testing=True,
    ):
        super(Options, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.mlp_dr = mlp_dr
        self.lr = lr
        self.tune_last_layer_only = tune_last_layer_only
        self.schedule_milestone = schedule_milestone
        self.gpus = gpus
        self.optimizer = optimizer
        self.gfn_dropout = gfn_dropout
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.mask = mask
        self.sample_num = sample_num
        self.t_test = t_test
        self.test_sample_mode = test_sample_mode
        self.use_t_in_testing = use_t_in_testing
