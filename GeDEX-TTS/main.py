import os
import sys

import warnings
warnings.filterwarnings('ignore')

from config import *
from text.symbols import symbols
from model.utils import fix_len_compatibility
from src.utils import *
from src.dataset import *
from src.train import *
from src.evaluation import *
from argument import *

def main(cfg):
    
    seed_init(seed=cfg.seed)
    if args.action == 'train':

        print('--- Train Phase ---')
        batch_collate = TextMelSpeakerBatchCollate()
        
        train_dataset = TextMelSpeakerDataset(cfg.path.train_path, cfg)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=cfg.train.batch_size, collate_fn=batch_collate, num_workers=cfg.num_worker, shuffle=True)
        
        val_dataset   = TextMelSpeakerDataset(cfg.path.val_path, cfg)
        val_loader    = DataLoader(dataset=val_dataset, batch_size=cfg.train.batch_size, collate_fn=batch_collate, num_workers=cfg.num_worker, shuffle=False)
        
        data_loader   = {'train':train_loader, 'valid':val_loader}

        trainer = Trainer(data_loader, cfg)
        trainer.train()

        if cfg.logging:
            neptune.stop()

    else:
        print('--- Test Phase ---')

        test(cfg, sample_size=10)


if __name__ == "__main__":
    
    args = get_config()
    cfg  = Config(args.config)
    cfg  = set_experiment(args, cfg)
    if args.action == 'train':
        shutil.copyfile(args.config, os.path.join(cfg.checkpoint, 'base.yaml'))
    else:
        cfg  = Config(os.path.join(cfg.checkpoint, 'base.yaml'))
        cfg  = set_experiment(args, cfg)
    
    cfg.model.n_vocab  = len(symbols) + 1 if cfg.model.add_blank else len(symbols)
    if cfg.train.out_size:
        cfg.train.out_size = fix_len_compatibility(cfg.train.fix_len * cfg.preprocess.sample_rate // cfg.preprocess.hop_length) 
    else:
        cfg.train.out_size = None
    print(cfg)
   
    main(cfg)
