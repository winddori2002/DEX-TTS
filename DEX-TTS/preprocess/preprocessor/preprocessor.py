import os
import random
import json

import tgt
import librosa
import soundfile as sf
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.io import wavfile

import sys
sys.path.append(os.path.abspath('./'))

import audio as Audio
random.seed(1234)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "trim_wav")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "lf0")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)

        print("Processing Data ...")
        out_list = []
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                out      = self.process_utterance(speaker, basename)
                out_list.append(out)

############################################
    def process_utterance(self, speaker, basename):
        wav_path  = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        ########################################################################
        """
        If you use mfa for trim wavs, you can use the below code.
        Empirically, it helps the network to converge faster and better.
        We used MFA recipes from https://github.com/ming024/FastSpeech2.
        We adopted the recipes for VCTK and ESD datasets.
        """
        # tg_path = os.path.join(
        #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        # )

        # # Get alignments
        # textgrid = tgt.io.read_textgrid(tg_path)
        # phone, duration, start, end = self.get_alignment(
        #     textgrid.get_tier_by_name("phones")
        # )
        # text = "{" + " ".join(phone) + "}"
        # if start >= end:
        #     return None

        # # Read and trim wav files
        # wav, _ = librosa.load(wav_path)
        # wav = wav[
        #     int(self.sampling_rate * start) : int(self.sampling_rate * end)
        # ].astype(np.float32)
        ########################################################################

        # Read and trim wav files   
        wav, _ = librosa.load(wav_path)
        wav    = wav.astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.STFT)
        
        # Save files
##################################################################
        wav_filename = "{}-wav-{}.wav".format(speaker, basename)
        sf.write(os.path.join(self.out_dir, "trim_wav", wav_filename), wav, self.sampling_rate)  
        
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "mel", mel_filename), mel_spectrogram.T)
        
        wav_filename = "{}-wav-{}.wav".format(speaker, basename)
        wav_path     = os.path.join(self.out_dir, "trim_wav", wav_filename)
        
        wav, fs = sf.read(wav_path)
        if fs != self.sampling_rate:
            print('--- Check wav sample rate ---')
        tlen         = mel_spectrogram.shape[-1]
        frame_period = self.hop_length / self.sampling_rate * 1000
        f0, timeaxis = pw.dio(wav.astype('float64'), self.sampling_rate, frame_period=frame_period)
        f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, self.sampling_rate)
        f0           = f0[:tlen].reshape(-1).astype('float32')
        
        nonzeros_indices      = np.nonzero(f0)
        lf0                   = f0.copy()
        lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
        
        lf0_filename = "{}-lf0-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "lf0", lf0_filename), lf0)
        
##################################################################
        return "|".join([basename, speaker, raw_text])