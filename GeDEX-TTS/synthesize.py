import os
import warnings
warnings.filterwarnings('ignore')

import argparse
from src.utils import *
from text import text_to_sequence, cmudict
from text.symbols import symbols
from model import GeDEXTTS
from scipy.io.wavfile import write
import numpy as np

global MAX_VALUE  
MAX_VALUE= 32768.0

def main(cfg):
    
    seed_init(seed=cfg.seed)   
    model = GeDEXTTS(cfg.model).to(cfg.device)
    ckpt  = torch.load(os.path.join(cfg.weight_path, 'model-train-best.pth'), map_location=cfg.device)
    if cfg.test.ema:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['state_dict'])

    vocoder = get_vocoder(cfg)
    text    = cfg.input_text
    print(text)
    cmu     = cmudict.CMUDict('./resources/cmu_dictionary')

    model.eval()
    x           = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(cfg.device)[None]
    x_lengths   = torch.LongTensor([x.shape[-1]]).to(cfg.device)
    spk         = torch.LongTensor([cfg.spk_id]).to(cfg.device)

    with torch.no_grad():
        y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=cfg.n_timesteps, temperature=1.5, spk=spk)
        audio = (vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * MAX_VALUE).astype(np.int16)

    basename    = str(spk.cpu().numpy()[0])
    output_name = basename + '_syn.wav'
    output_path = os.path.join(cfg.wav_path, output_name)

    write(output_path, 22050, audio)
    print('Done. Check out `out` folder for samples.')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',       type=str,   default='cuda:0', help='Device')
    parser.add_argument('--wav_path',     type=str,   default='./syn_samples', help='synthesize path')
    parser.add_argument('--spk_id',       type=int,   default=0, help='Pre-defined speaker ID')
    parser.add_argument('--weight_path',  type=str,   default='./checkpoints/GeDEX-TTS-LJSpeech', help='pre-trained weight path')
    parser.add_argument('--input_text',   type=str,   default='This is the test sentence.', help='input text')
    parser.add_argument('--seed',         type=int,   default=100, help='seed number')
    parser.add_argument('--n_timesteps',  type=int,   default=50, help='Time step')
    parser.add_argument('--temperature',  type=float, default=1.5, help='Temperature')
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
