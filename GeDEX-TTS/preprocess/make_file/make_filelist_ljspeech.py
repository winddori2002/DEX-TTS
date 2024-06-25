import os
import random
import argparse
import yaml
import numpy as np

import sys
sys.path.append(os.path.abspath('./'))
from src.utils import seed_init, MakeDir, parse_filelist, Config


def split_train_val_test(write_path, mel_path):
    
    ## We use dummy path --> test settings used in Grad-TTS and Glow-TTS 
    ref_path  = './resources/filelists/LJSpeech'
    ref_files =  os.listdir(ref_path)
    
    text_list = []
    for file in ref_files:
        with open(f"{ref_path}/{file}", encoding='utf-8') as f:
            strings = f.readlines()
        
        for i, line in enumerate(strings):
            strings[i] = line.replace('DUMMY', mel_path)
            txt        = line.split('|')[1]
            text_list.append(txt + '\n')
            
        f.close()
        
        with open(f"{write_path}/{file}", "w") as file:
            file.writelines(strings)
    
    text_list = sorted(list(set(text_list)))    
    random.shuffle(text_list)
        
    with open("test_sentence/ljspeech_sentence.txt", "w", encoding="utf-8") as file:
        file.writelines(text_list)
           
    
def main(cfg):
    
    seed_init(seed=100)
    
    write_path = f'./filelists/{cfg.dataset}'
    wav_path   = cfg.path.raw_path
    mel_path   = f'{cfg.path.preprocessed_path}/mel'
    MakeDir(write_path)
    
    ### Save filelist ####
    split_train_val_test(write_path, mel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/LJSpeech/preprocess.yaml")
    args = parser.parse_args()

    cfg = Config(args.config)

    main(cfg)
    