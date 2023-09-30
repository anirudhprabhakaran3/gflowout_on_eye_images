# Configuration for the project


class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.IMG_SIZE = 224
        self.BATCH_SIZE = 32
        self.N_CLASSES = 2
        self.N_EPOCHS = 10
        self.LEARNING_RATE = 0.01
        self.WEIGHT_DECAY = 0.001
        self.MOMENTUM = 0.9

        self.gfn_dropout = True
        self.lr = 1e-3
        self.lr_decay = 0.2
        self.use_pretrained = True
        self.tune_last_layer_only = True
        self.schedule_milestone = []
        self.dropout_rate = 0.5
        self.mlp_dropout_rate = 0.5
        self.gfn_model_training = True
        self.seed = 0
        self.optimizer = "adam"
        self.mask = "none"


config = Config()
