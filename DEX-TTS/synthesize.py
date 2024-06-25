import os

import warnings
warnings.filterwarnings('ignore')

import argparse
from tqdm import tqdm
from src.utils import *
from text import text_to_sequence, cmudict
from text.symbols import symbols
from model import DeXTTS

import soundfile as sf
import resampy
import audio as Audio
import pyworld as pw
from scipy.io.wavfile import write
import librosa
import numpy as np


global MAX_VALUE  
MAX_VALUE= 32768.0


def normalize_lf0(lf0):      
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

def preprocess_wav(path, STFT, cfg):
    
    # Read and trim wav files
    wav, fs = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=30) # trim slience
    if fs != cfg.preprocess.sample_rate:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=cfg.preprocess.sample_rate, axis=0)
    wav = wav / max(abs(wav))  
    
    mel, _ = Audio.tools.get_mel_from_wav(wav, STFT)

    tlen         = mel.shape[-1]
    frame_period = cfg.preprocess.hop_length / cfg.preprocess.sample_rate * 1000
    f0, timeaxis = pw.dio(wav.astype('float64'), cfg.preprocess.sample_rate, frame_period=frame_period)
    f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, cfg.preprocess.sample_rate)
    f0           = f0[:tlen].reshape(-1).astype('float32')
    
    nonzeros_indices      = np.nonzero(f0)
    lf0                   = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    lf0                   = normalize_lf0(lf0)

    return mel, lf0
    
def main(cfg):
    
    seed_init(seed=cfg.seed)   
    model = DeXTTS(cfg.model).to(cfg.device)
    ckpt  = torch.load(os.path.join(cfg.weight_path, 'model-train-best.pth'), map_location=cfg.device)
    if cfg.test.ema:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['state_dict'])

    vocoder = get_vocoder(cfg)
    text    = cfg.input_text
    print(text)
    cmu     = cmudict.CMUDict('./resources/cmu_dictionary')

    STFT = Audio.stft.TacotronSTFT(cfg.preprocess.n_fft,
                                   cfg.preprocess.hop_length,
                                   cfg.preprocess.win_length,
                                   cfg.preprocess.n_mels,
                                   cfg.preprocess.sample_rate,
                                   cfg.preprocess.f_min,
                                   cfg.preprocess.f_max)

    model.eval()
    for i, ref_name in enumerate(tqdm(cfg.ref_name)):
        
        ref_path = os.path.join(cfg.wav_path, ref_name)
        mel, lf0 = preprocess_wav(ref_path, STFT, cfg) 

        x           = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(cfg.device)[None]
        x_lengths   = torch.LongTensor([x.shape[-1]]).to(cfg.device)

        ref         = torch.Tensor(mel).unsqueeze(0).to(cfg.device)
        ref_lengths = torch.LongTensor([ref.shape[-1]]).to(cfg.device)
        sty         = torch.Tensor(mel).unsqueeze(0).to(cfg.device)
        sty_lengths = torch.LongTensor([sty.shape[-1]]).to(cfg.device)

        lf0         = torch.Tensor(lf0).unsqueeze(0).to(cfg.device)
        lf0_lengths = torch.LongTensor([lf0.shape[-1]]).to(cfg.device)

        with torch.no_grad():
            y_enc, y_dec, attn = model(x, x_lengths, ref, ref_lengths, sty, sty_lengths, lf0, lf0_lengths, spk=None, n_timesteps=cfg.n_timesteps, temperature=1.5)
            audio = (vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * MAX_VALUE).astype(np.int16)

        basename    = ref_name.split('.')[0]
        output_name = basename + '_syn.wav'
        output_path = os.path.join(cfg.wav_path, output_name)

        write(output_path, 22050, audio)
    print('Done. Check out `out` folder for samples.')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--wav_path', type=str, default='./syn_samples', help='synthesize path')
    parser.add_argument('--ref_name', type=str, nargs='+', default=['sample1.wav'], help='referecne audio name')
    parser.add_argument('--weight_path', type=str, default='./checkpoints/DEX-TTS-VCTK', help='pre-trained weight path')
    parser.add_argument('--input_text', type=str, default='This is the test sentence.', help='input text')
    parser.add_argument('--seed', type=int, default=100, help='seed number')
    parser.add_argument('--n_timesteps', type=int, default=50, help='Time step')
    parser.add_argument('--temperature', type=float, default=1.5, help='Temperature')
    parser.add_argument('--length_scale', type=float, default=1.0, help='length scale')
    
    args = parser.parse_args()
    
    cfg_path = os.path.join(args.weight_path, 'base.yaml')
    cfg  = Config(cfg_path)
    args = get_params(args)
    for key in args:
        cfg[key] = args[key]    
    cfg.model.n_vocab  = len(symbols) + 1 if cfg.model.add_blank else len(symbols)

    print(cfg)
    main(cfg)
