import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader 

from text import text_to_sequence, cmudict
from text.symbols import symbols
from src.utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility

class TextMelSpeakerDataset(Dataset):
    def __init__(self, filelist_path, cfg):
        super().__init__()
        self.filelist    = parse_filelist(filelist_path, split_char='|')
        self.cmudict     = cmudict.CMUDict(cfg.path.cmu_path)
        self.add_blank   = cfg.model.add_blank
        
        random.seed(cfg.seed)
        random.shuffle(self.filelist)
        # self.filelist = self.filelist[:100]
        
    def __len__(self):
        return len(self.filelist)
    
    def get_data(self, line):
        
        filepath, text, speaker = line[0], line[1], line[2]
        text    = self.get_text(text, add_blank=self.add_blank)
        mel     = self.get_mel_from_path(filepath)
        speaker = self.get_speaker(speaker)
        
        return (text, mel, speaker)
    
    def get_mel_from_path(self, filepath):
        mel = torch.Tensor(np.load(filepath).T)
        return mel

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
        speaker = self.get_speaker(speaker)
        if len(emotion) != 0:
            emotion = emotion[0]
        else:
            emotion = 'None'
        item = {'y': mel, 'x': text, 'spk': speaker, 'filepath':filepath, 'emotion':emotion, 'raw_text': raw_text}
        return item

    def __getitem__(self, index):
        
        text, mel, speaker = self.get_data(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker}
        
        return item

class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        
        B            = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats      = batch[0]['y'].shape[-2]

        y   = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x   = torch.zeros((B, x_max_length), dtype=torch.long)
        spk = []
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            spk.append(spk_)
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]]    = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk       = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}
