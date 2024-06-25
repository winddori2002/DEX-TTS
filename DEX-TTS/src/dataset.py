import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader 

from text import text_to_sequence, cmudict
from text.symbols import symbols
from src.utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from model.augmentation import Augment

class TextMelSpeakerDataset(Dataset):
    def __init__(self, filelist_path, cfg):
        super().__init__()
        self.filelist    = parse_filelist(filelist_path, split_char='|')
        self.cmudict     = cmudict.CMUDict(cfg.path.cmu_path)
        self.add_blank   = cfg.model.add_blank
        self.ref_type    = cfg.train.ref_type
        self.sty_type    = cfg.train.sty_type
        self.aug_type    = cfg.train.aug_type
        self.augment     = Augment()
        
        self.ref_tm      = 13 if self.ref_type != 'mel' else 27
        self.sty_tm      = 13 if self.sty_type != 'mel' else 27
        self.fm          = 50

        random.seed(cfg.seed)
        random.shuffle(self.filelist)
        # self.filelist = self.filelist[:100]
                
    def __len__(self):
        return len(self.filelist)
    
    def get_data(self, line):
        
        filepath, text, speaker = line[0], line[1], line[2]

        text    = self.get_text(text, add_blank=self.add_blank)
        mel     = self.get_mel_from_path(filepath)
        ref     = self.get_feat_from_path(filepath, self.ref_type)
        sty     = self.get_feat_from_path(filepath, self.sty_type)
        lf0     = self.get_lf0_from_path(filepath)
        speaker = self.get_speaker(speaker)
        
        return (text, mel, ref, sty, lf0, speaker)
    
    def get_feat_from_path(self, filepath, feat_type):
        if feat_type == 'mel':
            ref = torch.Tensor(np.load(filepath).T)
        return ref
        
    def get_mel_from_path(self, filepath):
        mel = torch.Tensor(np.load(filepath).T)
        return mel
    
    def normalize_lf0(self, lf0):      
        zero_idxs    = np.where(lf0 == 0)[0]
        nonzero_idxs = np.where(lf0 != 0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std  = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0
        return lf0
    
    def get_lf0_from_path(self, filepath):
        filepath = filepath.replace('/mel/', '/lf0/').replace('-mel-', '-lf0-')
        lf0      = np.load(filepath)
        lf0      = self.normalize_lf0(lf0)
        lf0      = torch.Tensor(lf0)
        return lf0
    
    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker
    
    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch
    
    def get_sample_idx(self, spk_list):
        idx_list = []
        for idx, line in enumerate(self.filelist):
            filepath, text, speaker = line[0], line[1], line[2]
            if int(speaker) in spk_list:
                idx_list.append(idx)
        return idx_list

    def get_eval_data(self, index):
        line = self.filelist[index]
        filepath, raw_text, speaker, *emotion = line
        text    = self.get_text(raw_text, add_blank=self.add_blank)
        mel     = self.get_mel_from_path(filepath)
        ref     = self.get_feat_from_path(filepath, self.ref_type)
        sty     = self.get_feat_from_path(filepath, self.sty_type)
        lf0     = self.get_lf0_from_path(filepath)
        speaker = self.get_speaker(speaker)
        
        if len(emotion) != 0:
            emotion = emotion[0]
        else:
            emotion = 'None'
        item = {'y': mel, 'x': text, 'ref': ref, 'sty':sty, 'lf0':lf0, 'spk': speaker, 'filepath':filepath, 'emotion':emotion, 'raw_text': raw_text}
        return item
    
    def __getitem__(self, index):
        
        text, mel, ref, sty, lf0, speaker = self.get_data(self.filelist[index])        
            
        ref     = self.augment(ref, aug_type=self.aug_type[0], time_mask_para=self.ref_tm, freq_mask_para=self.fm)
        lf0     = self.augment(lf0, aug_type=self.aug_type[1], time_mask_para=27, freq_mask_para=50)
        sty     = self.augment(sty, aug_type=self.aug_type[2], time_mask_para=self.sty_tm, freq_mask_para=self.fm)
        
        ## ref --> TIV, sty --> TV (we name the variable not to be confused)
        item = {'y': mel, 'x': text, 'ref':ref, 'sty':sty, 'lf0':lf0, 'spk': speaker}
        
        return item

class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        
        B            = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats      = batch[0]['y'].shape[-2]
        
        ref_feats      = batch[0]['ref'].shape[-2]
        ref_max_length = max([item['ref'].shape[-1] for item in batch])
        sty_feats      = batch[0]['sty'].shape[-2]
        sty_max_length = max([item['sty'].shape[-1] for item in batch])
        lf0_max_length = max([item['lf0'].shape[-1] for item in batch])

        y   = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x   = torch.zeros((B, x_max_length), dtype=torch.long)
        ref = torch.zeros((B, ref_feats, ref_max_length), dtype=torch.float32)
        sty = torch.zeros((B, sty_feats, sty_max_length), dtype=torch.float32)
        lf0 = torch.zeros((B, lf0_max_length), dtype=torch.float32)
        spk = []
        y_lengths, x_lengths, ref_lengths, sty_lengths, lf0_lengths = [], [], [], [], []

        for i, item in enumerate(batch):
            y_, x_, spk_, ref_, sty_, lf0_ = item['y'], item['x'], item['spk'], item['ref'], item['sty'], item['lf0']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            ref_lengths.append(ref_.shape[-1])
            sty_lengths.append(sty_.shape[-1])
            lf0_lengths.append(lf0_.shape[-1])
            spk.append(spk_)
            y[i, :, :y_.shape[-1]]     = y_
            x[i, :x_.shape[-1]]        = x_
            ref[i, :, :ref_.shape[-1]] = ref_
            sty[i, :, :sty_.shape[-1]] = sty_
            lf0[i, :lf0_.shape[-1]]    = lf0_

        y_lengths   = torch.LongTensor(y_lengths)
        x_lengths   = torch.LongTensor(x_lengths)
        ref_lengths = torch.LongTensor(ref_lengths)
        sty_lengths = torch.LongTensor(sty_lengths)
        lf0_lengths = torch.LongTensor(lf0_lengths)
        spk         = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk, 'ref':ref, 'ref_lengths':ref_lengths, 'sty':sty, 'sty_lengths':sty_lengths, 'lf0':lf0, 'lf0_lengths':lf0_lengths}
