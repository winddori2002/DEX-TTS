import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

import soundfile as sf
from text import _clean_text
import resampy

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "LJSpeech"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                                
                wav, fs = sf.read(wav_path)
                if fs != sampling_rate:
                    wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sampling_rate, axis=0)
                wav = wav / max(abs(wav))
                sf.write(os.path.join(out_dir, speaker, "{}.wav".format(base_name)), wav, sampling_rate)       
                
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
                