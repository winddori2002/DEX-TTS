import numpy as np
import os
import torch
import torchaudio
import torch.nn as nn

from scipy.io.wavfile import write
import neptune

import datetime as dt
from tqdm import tqdm
from src.dataset import *
from src.utils import *
from src.metric import *
from model import DeXTTS
import shutil

MAX_VALUE  = 32768.0

def test(cfg, sample_size):
    
    syn_path = f'{cfg.eval_path}/syn/'
    ref_path = f'{cfg.eval_path}/ref/'
    MakeDir(syn_path)
    MakeDir(ref_path)

    seed_init(seed=cfg.seed)   
    model = DeXTTS(cfg.model).to(cfg.device)
    ckpt  = torch.load(f'{cfg.checkpoint}/model-train-best.pth', map_location=cfg.device)
    if cfg.test.ema:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['state_dict'])

    vocoder   = get_vocoder(cfg)
    evaluater = Evaluater(cfg) 
    
    with open(cfg.test_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    texts = texts[:sample_size]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    test_dataset = TextMelSpeakerDataset(cfg.path.test_path, cfg)

    model.eval()
    meta_data =  {'trg_path':[], 'syn_path':[], 'text':[]}
    for i, text in enumerate(tqdm(texts)):
        
        item = test_dataset.get_eval_data(i)
        if cfg.pa:
            x    = item['x'].unsqueeze(0).to(cfg.device)
            text = item['raw_text']
        else:
            x    = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(cfg.device)[None]

        x_lengths   = torch.LongTensor([x.shape[-1]]).to(cfg.device)
        y           = item['y'].unsqueeze(0).to(cfg.device)
        ref         = item['ref'].unsqueeze(0).to(cfg.device)
        ref_lengths = torch.LongTensor([ref.shape[-1]]).to(cfg.device)
        sty         = item['sty'].unsqueeze(0).to(cfg.device)
        sty_lengths = torch.LongTensor([sty.shape[-1]]).to(cfg.device)
        lf0         = item['lf0'].unsqueeze(0).to(cfg.device)
        lf0_lengths = torch.LongTensor([lf0.shape[-1]]).to(cfg.device)
        spk         = item['spk'].item()
        
        wav_path    = item['filepath'].replace('/mel/','/trim_wav/').replace('-mel-', '-wav-').replace('.npy', '.wav')
        emotion     = item['emotion']

        with torch.no_grad():
            y_enc, y_dec, attn = model(x, x_lengths, ref, ref_lengths, sty, sty_lengths, lf0, lf0_lengths, spk=None, n_timesteps=cfg.n_timesteps, temperature=1.5)
            audio = (vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * MAX_VALUE).astype(np.int16)

        syn_save_path  = f'{cfg.eval_path}/syn/spk_{spk}_{i}_{emotion}.wav'
        wav_save_path  = f'{cfg.eval_path}/ref/spk_{spk}_{i}_{emotion}.wav'

        write(syn_save_path, 22050, audio)
        shutil.copyfile(wav_path, wav_save_path)
        meta_data['trg_path'].append(wav_save_path)
        meta_data['syn_path'].append(syn_save_path)
        meta_data['text'].append(text)
        
    cer, cer_std, wer, wer_std = evaluater.calculate_asr_score(meta_data)
    cos, cos_std = evaluater.calculate_asv_score(meta_data)
    
    msg = 'Seen Test results | CER:{:.4f} | WER:{:.4f} | COS:{:.4f} \n'.format(cer, wer, cos)
    print(msg)
    
class Tester:
    def __init__(self, cfg):
        
        self.cfg       = cfg
        self.model     = DeXTTS(cfg.model).to(cfg.device)
        self.vocoder   = get_vocoder(cfg)
        self.dataset   = TextMelSpeakerDataset(cfg.path.val_path, cfg)

    def _save_log(self, msg):
        with open(f'{self.cfg.checkpoint}/log.txt', 'a') as f:
            f.write(msg)
        
    def synthesize(self, sample_size=100):
        
        syn_path = f'{self.cfg.sample_path}/syn/'
        ref_path = f'{self.cfg.sample_path}/ref/'
        MakeDir(syn_path)
        MakeDir(ref_path)
        
        checkpoint = torch.load(f'{self.cfg.checkpoint}/model-train-best.pth', map_location=self.cfg.device)
        if self.cfg.test.ema:
            self.model.load_state_dict(checkpoint['ema'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        
        with open(self.cfg.test_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        texts = texts[:sample_size]
        cmu = cmudict.CMUDict('./resources/cmu_dictionary')
        
        self.model.eval()
        for i, text in enumerate(tqdm(texts)):
            
            item = self.dataset.get_eval_data(i)
            if self.cfg.pa:
                x    = item['x'].unsqueeze(0).to(self.cfg.device)
                text = item['raw_text']
            else:
                x    = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(self.cfg.device)[None]

            x_lengths   = torch.LongTensor([x.shape[-1]]).to(self.cfg.device)
            y           = item['y'].unsqueeze(0).to(self.cfg.device)
            ref         = item['ref'].unsqueeze(0).to(self.cfg.device)
            ref_lengths = torch.LongTensor([ref.shape[-1]]).to(self.cfg.device)
            sty         = item['sty'].unsqueeze(0).to(self.cfg.device)
            sty_lengths = torch.LongTensor([sty.shape[-1]]).to(self.cfg.device)
            lf0         = item['lf0'].unsqueeze(0).to(self.cfg.device)
            lf0_lengths = torch.LongTensor([lf0.shape[-1]]).to(self.cfg.device)
            spk         = item['spk'].item()
            
            wav_path    = item['filepath'].replace('/mel/','/trim_wav/').replace('-mel-', '-wav-').replace('.npy', '.wav')
            emotion     = item['emotion']

            with torch.no_grad():
                y_enc, y_dec, attn = self.model(x, x_lengths, ref, ref_lengths, sty, sty_lengths, lf0, lf0_lengths, spk=None, n_timesteps=self.cfg.n_timesteps, temperature=1.5)
                audio = (self.vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * MAX_VALUE).astype(np.int16)

            syn_save_path  = f'{self.cfg.sample_path}/syn/spk_{spk}_{i}_{emotion}.wav'
            wav_save_path  = f'{self.cfg.sample_path}/ref/spk_{spk}_{i}_{emotion}.wav'

            write(syn_save_path, 22050, audio)
            shutil.copyfile(wav_path, wav_save_path)



