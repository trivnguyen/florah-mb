
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.enable_progress_bar = True

    config.seed_data = 10
    config.seed_training = 11
    config.workdir = '/home/ncn2112/trained-models'
    config.name = 'sfh5-testrun'
    config.overwrite = False
    config.checkpoint = 'last-epoch=3225-step=100000-val_loss=-6.7745.ckpt'

    config.data = data = ConfigDict()
    data.root = '/home/ncn2112/sfh_dataset'
    data.name = 'sfh5'
    data.features = ['sfr', 'mstar']
    data.time_features = ['aexp', ]
    data.reverse_time = False

    config.training = training = ConfigDict()
    training.train_batch_size = 32
    training.eval_batch_size = 32
    training.train_frac = 0.8
    training.num_workers = 0
    training.max_steps = 200_000
    training.accelerator = 'gpu'
    training.gradient_clip_val = 0.5
    training.save_top_k = 3

    # optimizer and scheduler configuration
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 5e-4
    optimizer.betas = (0.9, 0.98)
    optimizer.weight_decay = 1e-4
    optimizer.eps = 1e-9
    config.scheduler = scheduler = ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    scheduler.decay_steps = 100_000  # include warmup steps
    scheduler.warmup_steps = 10_000
    scheduler.eta_min = 0.01
    scheduler.interval = 'step'

    # model configuration
    config.model = model = ConfigDict()
    model.d_in = len(data.features)
    model.d_time = len(data.time_features) * 2
    config.model.encoder = encoder = ConfigDict()
    encoder.name = 'transformer'
    encoder.d_model = 128
    encoder.nhead = 4
    encoder.dim_feedforward = 256
    encoder.num_layers = 2
    encoder.emb_size = 64
    encoder.emb_dropout = 0.1
    encoder.emb_type = 'linear'
    encoder.concat = False
    config.model.npe = npe = ConfigDict()
    npe.hidden_sizes = [128, 128]
    npe.context_embedding_sizes = [128, 128]
    npe.num_transforms = 4
    npe.dropout = 0.1

    return config

config = get_config()