import os
import random
import argparse
import yaml
import numpy as np

import librosa
import sys
sys.path.append(os.path.abspath('./'))
from src.utils import seed_init, MakeDir, parse_filelist, Config

def filter_duration(write_path, raw_wav_path, write_filename, min_dur, max_dur):
    
    filtered_list = []
    total_spks    = []
    filtered_spks = []
    total_dur     = 0
    filtered_dur  = 0
    with open(f"{write_path}/{write_filename}", "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            line_path, text, speaker = line.strip("\n").split("|")
            
            basename = line_path.split('-')[-1].replace('npy', 'wav')
            
            wav_path = f'{raw_wav_path}/{speaker}/{basename}'
            wav, sr  = librosa.load(wav_path)
            duration = wav.shape[0] / sr
            
            total_spks.append(speaker)
            total_dur += duration
            
            if duration <= max_dur and duration >= min_dur:
                filtered_list.append(line)
                filtered_spks.append(speaker)
                filtered_dur += duration
            
    total_spks    = set(total_spks)
    filtered_spks = set(filtered_spks)

    print(f'Num files: {i+1} ---> {len(filtered_list)}')
    print(f'Num spks: {len(total_spks)} ---> {len(filtered_spks)}')
    print(f'Duration: {total_dur} ---> {filtered_dur}')
    print(f'Duration: {total_dur/3600}h ---> {filtered_dur/3600}h')

    filtered_list = sorted(filtered_list)
    random.shuffle(filtered_list)
    
    new_write_filename = write_filename.split('.')[0] + '-' + str(min_dur) + '-' + str(max_dur) + '.txt'
    with open(f"{write_path}/{new_write_filename}", "w") as file:
        file.writelines(filtered_list)

    
def save_filelist(write_path, wav_path, mel_path, write_filename):
    
    mel_files = os.listdir(mel_path)
    
    file_list = []
    for file in mel_files:
        spk, filename = file.split('-')[0], file.split('-')[-1]
        text_path     = os.path.join(wav_path, spk, filename.replace('.npy','.lab'))
        
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        file_path = os.path.join(mel_path, file)
        strings   = file_path + '|' + raw_text + '|' + spk + '\n'
        file_list.append(strings)
        

    file_list = sorted(file_list)
    random.shuffle(file_list)

    print('Number of files:', len(file_list))
    with open(f"{write_path}/{write_filename}", "w") as file:
        file.writelines(file_list)
    

def main(cfg, args):
    
    seed_init(seed=100)
    
    write_path = f'./filelists/{cfg.dataset}'
    wav_path   = cfg.path.raw_path
    mel_path   = f'{cfg.path.preprocessed_path}/mel'
    MakeDir(write_path)

    spk_list = sorted(os.listdir(wav_path))
    print('Number of speakers:', len(spk_list))

    
    ### Save filelist ####
    save_filelist(write_path, wav_path, mel_path, args.filename)
    
    
    ### filter fileslist depending on audio duration ####
    filter_duration(write_path, wav_path, args.filename, args.min_dur, args.max_dur)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/LibriTTS/preprocess.yaml")
    parser.add_argument("--filename", type=str, default="test-clean.txt")
    parser.add_argument("--min_dur", type=int, default=4)
    parser.add_argument("--max_dur", type=int, default=10)
    args = parser.parse_args()

    cfg = Config(args.config)
    
    main(cfg, args)
    