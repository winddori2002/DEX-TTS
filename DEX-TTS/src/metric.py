import logging
import librosa
import numpy as np

from tqdm import tqdm
from typing import Optional, Union
from pathlib import Path
from src.utils import *

import jiwer
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
from resemblyzer import VoiceEncoder, normalize_volume, trim_long_silences

class Evaluater:
    
    def __init__(self, cfg):
        
        self.cfg       = cfg
        self.asv_model = VoiceEncoder(device=cfg.device)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(cfg.device)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        
    def transcribe(self, wav_path):
        
        wav, _         = librosa.load(wav_path, sr=16000)
        inputs         = self.tokenizer(wav, sampling_rate=16000, return_tensors="pt", padding="longest")
        input_values   = inputs.input_values.to(self.cfg.device)
        attention_mask = inputs.attention_mask.to(self.cfg.device)

        logits        = self.asr_model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription
        
    def calculate_wer_cer(self, gt, pred):
        
        gt   = normalize_sentence(gt)
        pred = normalize_sentence(pred)
        cer  = jiwer.cer(gt, pred)
        wer  = jiwer.wer(gt, pred)     
        
        return cer, wer
        
    def calculate_asr_score(self, meta_data):
        
        cer_list  = []
        wer_list  = []
        total_cer = 0
        total_wer = 0
        trg_paths = meta_data['trg_path']
        syn_paths = meta_data['syn_path']
        texts     = meta_data['text']
        for (trg, syn, txt) in zip(trg_paths, syn_paths, texts):
                        
            pred      = self.transcribe(syn)
            cer, wer  = self.calculate_wer_cer(txt, pred)
            total_cer += cer
            total_wer += wer        
            cer_list.append(cer)
            wer_list.append(wer)

        cer_std = np.std(cer_list) / np.sqrt(len(trg_paths))
        wer_std = np.std(wer_list) / np.sqrt(len(trg_paths))
        
        return total_cer / len(trg_paths), cer_std, total_wer / len(trg_paths), wer_std
    
    def calculate_accept(self, cnv, trg):
        
        cnv     = preprocess_wav(cnv, source_sr=self.cfg.preprocess.sample_rate)
        cnv_emb = self.asv_model.embed_utterance(cnv)

        trg     = preprocess_wav(trg, source_sr=self.cfg.preprocess.sample_rate)
        trg_emb = self.asv_model.embed_utterance(trg)

        cos = np.inner(cnv_emb, trg_emb) / (np.linalg.norm(cnv_emb) * np.linalg.norm(trg_emb))

        return cos
    
    def calculate_asv_score(self, meta_data):
        
        cos_sum   = 0
        cos_list  = []
        trg_paths = meta_data['trg_path']
        syn_paths = meta_data['syn_path']
        texts     = meta_data['text']
        for (trg, syn, txt) in zip(trg_paths, syn_paths, texts):
            cos      = self.calculate_accept(syn, trg)
            cos_sum  += cos
            cos_list.append(cos)
            
        cos_std = np.std(cos_list) / np.sqrt(len(trg_paths))
            
        return cos_sum / len(trg_paths), cos_std

def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to upper.
    sentence = sentence.upper()

    return sentence

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int]=None):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that  
    The waveform will be resampled to match the data hyperparameters.
    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform'speaker sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav
    if source_sr is not None:
        sampling_rate = 16000
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    audio_norm_target_dBFS = -30
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav

