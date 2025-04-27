
from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # logging configuration
    config.workdir = '/mnt/ceph/users/tnguyen/florah-main-branch/logging/vsmdpl-dt1_4'
    config.name = 'gru'
    config.overwrite = True
    config.enable_progress_bar = False
    config.checkpoint = None
    config.accelerator = 'gpu'

    # seed
    config.seed = seed = config_dict.ConfigDict()
    seed.data = 5000
    seed.training = 5101

    config.data = data = config_dict.ConfigDict()
    data.root = "/mnt/ceph/users/tnguyen/florah-main-branch/datasets/processed"
    data.name = "vsmdpl-dt1_4"
    data.num_files = 10
    data.train_frac = 0.8
    data.reverse_time = False

    # model configuration
    config.model = model = config_dict.ConfigDict()
    model.d_in = 2
    model.num_classes= 3
    # transformer args
    model.encoder = encoder = config_dict.ConfigDict()
    model.encoder.name = 'gru'
    model.encoder.d_model = 32
    model.encoder.d_out = 32
    model.encoder.dim_feedforward = 32
    model.encoder.num_layers = 3
    model.encoder.concat = False
    # decoder args
    model.decoder = decoder = config_dict.ConfigDict()
    model.decoder.name = 'gru'
    model.decoder.d_model = 32
    model.decoder.d_out = 32
    model.decoder.dim_feedforward = 32
    model.decoder.num_layers = 3
    model.decoder.concat = False
    # npe args
    model.npe  = npe = config_dict.ConfigDict()
    model.npe.hidden_sizes = [32, 32]
    model.npe.num_transforms = 4
    model.npe.context_embedding_sizes = None
    model.npe.dropout = 0.0

    # optimizer and scheduler configuration
    config.optimizer = optimizer = config_dict.ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 5e-4
    optimizer.betas = (0.9, 0.98)
    optimizer.weight_decay = 1e-4
    optimizer.eps = 1e-9
    config.scheduler = scheduler = config_dict.ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    scheduler.decay_steps = 500_000  # include warmup steps
    scheduler.warmup_steps = 50_000
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'

    # training args
    config.training = training = config_dict.ConfigDict()
    training.max_epochs = 100_000
    training.max_steps = 500_000
    training.train_batch_size = 64
    training.eval_batch_size = 64
    training.monitor = 'val_loss'
    training.patience = 50
    training.save_top_k = 5
    training.save_last_k = 5
    training.gradient_clip_val = 0.5

    # model training args
    training.use_desc_mass_ratio = True

    return config
