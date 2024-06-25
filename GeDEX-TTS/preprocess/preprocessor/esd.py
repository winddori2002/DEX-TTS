import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

import fnmatch
from glob import glob
from tqdm import tqdm
from text import _clean_text
from g2p_en import G2p
import resampy
import soundfile as sf

def find_files(root_dir, query="*.wav", include_root_dir=True):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def prepare_align(config):
    in_dir  = config["path"]["corpus_path"] 
    txt_dir = in_dir.replace('wav48', 'txt')
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    base_speaker = '0011'
    base_text_path = os.path.join(in_dir, base_speaker, "{}.txt".format(base_speaker))
    meta_dict = {}
    with open(base_text_path, encoding='utf-8') as file:
        for line in file:
            if len(line) > 2:
                base_name, text, emotion = line.strip('\n').split('\t')
                text      = _clean_text(text, cleaners)
                key_name = base_name.replace(base_speaker+'_', '') # 0011_01234 -> 01234
                meta_dict[key_name] = [text, emotion] # 01234: [text, emotion]

    for speaker in tqdm(sorted(os.listdir(in_dir))):
        

        if '00' not in speaker:
            continue
        
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)   
        # # wav file list
        
        wav_path_list = find_files(os.path.join(in_dir, speaker))
        for wav_path in wav_path_list:
            base_name = os.path.basename(wav_path).split('.wav')[0] # speaker_01234
            
            ################### update version for removing weird sound #################
            wav, fs = sf.read(wav_path)
            if fs != sampling_rate:
                wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sampling_rate, axis=0)
            wav = wav / max(abs(wav))
            sf.write(os.path.join(out_dir, speaker, "{}.wav".format(base_name)), wav, sampling_rate)            

            key_name = base_name.replace(speaker+'_', '')
            text, emotion = meta_dict[key_name]
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w"
            ) as f1:
                f1.write(text)
                

def make_meta_dict(config):
    in_dir   = config["path"]["corpus_path"] 
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    base_speaker = '0011'
    base_text_path = os.path.join(in_dir, base_speaker, "{}.txt".format(base_speaker))
    meta_dict = {}
    with open(base_text_path, encoding='utf-8') as file:
        for line in file:
            if len(line) > 2:
                base_name, text, emotion = line.strip('\n').split('\t')
                text      = _clean_text(text, cleaners)
                key_name = base_name.replace(base_speaker+'_', '') # 0011_01234 -> 01234
                meta_dict[key_name] = [text, emotion]              # 01234: [text, emotion]

    np.save(f'{in_dir}/meta_dict.npy', meta_dict)