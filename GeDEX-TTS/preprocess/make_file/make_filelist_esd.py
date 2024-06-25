import os
import random
import argparse
import yaml
import numpy as np

import sys
sys.path.append(os.path.abspath('./'))
from src.utils import seed_init, MakeDir, parse_filelist, Config


def split_train_val_test(write_path, wav_path, mel_path, meta_path, spk_dict):
    
    meta_dic     = np.load(meta_path, allow_pickle=True).item()
    mel_filelist = os.listdir(mel_path)
    
    filelist  = []
    text_list = []
    for mel_file in mel_filelist:
        
        spk, basename = mel_file.split('-')[0], mel_file.split('-')[-1][:-4]  # remove .npy
        key_name  = basename.split('_')[-1]
        txt, emo  = meta_dic[key_name][0], meta_dic[key_name][1]
        spk       = str(spk_dict[spk])
        file_path = os.path.join(mel_path, mel_file)
        strings   = '|'.join([file_path, txt, spk, emo+'\n'])
        
        filelist.append(strings)
        text_list.append(txt + '\n')
        
    filelist = sorted(filelist)
    random.shuffle(filelist)
    
    val_size  = int(0.8 * len(filelist))
    test_size = int(0.9 * len(filelist))
    train_filelist = filelist[:val_size]
    val_filelist   = filelist[val_size:test_size]
    test_filelist  = filelist[test_size:]
    print(len(filelist), len(train_filelist), len(val_filelist), len(test_filelist))
    with open(f"{write_path}/train.txt", "w") as file:
        file.writelines(train_filelist)
    with open(f"{write_path}/valid.txt", "w") as file:
        file.writelines(val_filelist)
    with open(f"{write_path}/test.txt", "w") as file:
        file.writelines(test_filelist)
        
    ### get text
    
    text_list = sorted(list(set(text_list)))    
    random.shuffle(text_list)
        
    with open("test_sentence/esd_sentence.txt", "w", encoding="utf-8") as file:
        file.writelines(text_list)

def make_unseen_filelist(write_path, unseen_spk):

    for phase in ['train', 'valid']:
        filtered_list = []

        with open(os.path.join(write_path, f'{phase}.txt'), "r", encoding="utf-8") as f:
            
            strings = f.readlines()
            for i, line in enumerate(strings):
                mel_path, text, spk, *_ = line.strip("\n").split("|")
                
                if int(spk) in unseen_spk:
                    continue
                else:
                    filtered_list.append(line)        
        
        num_origin = len(strings)
        num_new    = len(filtered_list)

        with open(os.path.join(write_path, f"{phase}_unseen.txt"), "w", encoding="utf-8") as file:
            file.writelines(filtered_list)

        print(f'{phase} size: {num_origin} --> {num_new}')
    

def main(cfg):
    
    seed_init(seed=100)
    
    write_path = f'./filelists/{cfg.dataset}'
    wav_path   = cfg.path.raw_path
    mel_path   = f'{cfg.path.preprocessed_path}/mel'
    meta_path  = f'{cfg.path.raw_path}/meta_dict.npy'.replace('/raw_data', '')
    MakeDir(write_path)

    spk_list = sorted(os.listdir(wav_path))
    print('Number of speakers:', len(spk_list))
    spk_dict = {k:v for v,k in enumerate(spk_list)}
    print(spk_dict)
    
    unseen_spk = [0, 7]
    print('Unseen speaker:', unseen_spk)
    
    
    ### Save filelist ####
    split_train_val_test(write_path, wav_path, mel_path, meta_path, spk_dict)
        
    ### filter for unseen spekaer ####
    make_unseen_filelist(write_path, unseen_spk)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ESD/preprocess.yaml")
    args = parser.parse_args()

    cfg = Config(args.config)
    ## 0011: MAEL, 0018: FEMALE
    
    main(cfg)
    