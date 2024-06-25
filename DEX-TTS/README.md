# DEX-TTS: Diffusion-based EXpressive Text-to-Speech with Style Modeling on Time Variability

This repository is the official implementation of DEX-TTS: Diffusion-based EXpressive Text-to-Speech with Style Modeling on Time Variability. 

In this repository, we provide steps for running DEX-TTS. 

If you want to run GeDEX-TTS, move to [GeDEX-TTS](https://github.com/winddori2002/DEX-TTS/tree/main/GeDEX-TTS) repository.

🙏 We recommend you visit our [demo site](https://dextts.github.io/demo.github.io/). 🙏

DEX-TTS is diffusion-based expressive TTS which can extract and represent rich styles from the reference speech, using style modeling on time variability. The overall architecture of DEX-TTS is as below:

<p align="center">
	<img src="../images/DEXTTS.png" alt="DEX-TTS" width="80%" height="80%"/>
</p>

## Requirements

First install torch based on your environment (We tested on torch version (1.10.1, 1.12.1, 2.1.0)).

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

To install requirements:

```
pip install -r requirements.txt
```

For MAS algorithmm run the code below.

```
cd ./model/monotonic_align; python setup.py build_ext --inplace
```

For pre-trained [HiFi-GAN](https://github.com/jik876/hifi-gan) or [BigVGAN](https://github.com/NVIDIA/BigVGAN) vocoder, download vocoder [weights](https://github.com/winddori2002/DEX-TTS/releases/tag/weights), unzip them, and  place them ```./bigvgan``` or ```hifigan/weights```. 

## Prepare datasets

For expressive TTS, we use the VCTK and ESD datasets.

- The VCTK dataset can be downloaded [here](https://datashare.ed.ac.uk/handle/10283/2651).

- The ESD dataset can be downloaded [here](https://github.com/HLTSingapore/Emotional-Speech-Data?tab=readme-ov-file).

- For the ESD datatset, we only use Engligh speakers (0011 ~ 0020).

- We divide the datasets to design seen and unseen (zero-shot) scenarios.


## Preprocess

To generate features (mel-spectrograms and pitch), run the following code with the config option (VCTK or ESD):

```
python ./preprocess/preprocess.py --config ./config/{dataset}/preprocess.yaml

Example:
    python ./preprocess/preprocess.py --config ./config/VCTK/preprocess.yaml
```

To design zero-shot run the following codes:

```
python ./preprocess/make_file/make_filelist_esd.py
python ./preprocess/make_file/make_filelist_vctk.py
```

The codes yield the meta datalist for training with the format of ```Path|Text|Speaker|Emotion```.

We skipped the preprocess stage using [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/#). We used MFA recipes from [here](https://github.com/ming024/FastSpeech2). 

Empirically, trim silences using MFA helps the network to converge faster and better.


## Training

To train DEX-TTS from the scratch, run the following code.

If you want to change training options such as num_worker, cuda device, and so on, check ```argument.py```.

If you want to edit model or training settings, check ```config/{dataset}/base.yaml```. 

```train
python main.py train --config config/{dataset}/base.yaml

Configurations:
    ├── path
    ├── preprocess
    ├── model
    |     ├── dit
    │     ├── encoder
    │     ├── decoder
    │     ├── tv_encoder
    │     ├── tiv_encoder
    │     ├── lf0_encoder
    ├── train
    ├── test
```

## Evaluation

You can check Word Error Rate or Cosine Similarity of the synthesized samples by running the code.

```--pa``` indicates parallel options (text of reference speech and text input are the same).

You should match the checkpoint name for ```--test_checkpoint``` option.

```
python main.py test --config config/{dataset}/base.yaml --test_file [test_sentence files] --pa [True/None] --test_checkpoint [checkpoint name]

Example:
    python main.py test --config config/VCTK/base.yaml --test_file ./test_sentence/vctk_sentence --pa True --test_checkpoint 0
```


## Synthesize

If you want to synthesize samples, run the following codes with pre-trained models. 

```
python synthesize.py --input_text "" --wav_path ./syn_samples --ref_name {wav name} --weight_path {weight path}
```

## Pre-trained Models

Pre-trained models will be provided.

## ToDo
- [X] Bigvgan vocoder for multi-speaker TTS
- [ ] Multi-gpu training codes
- [ ] LibriTTS & Simpe preprocess recipes
- [ ] Pre-trained weight
- [ ] Precondition VE & VP
- [ ] Evaluation

## Citation

```

```

## License

> This repository will be released under the MIT license. 

> Thanks to the open source codebases such as [RetNet](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py), [FastSpeech2](https://github.com/ming024/FastSpeech2), [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS), [DiT](https://github.com/facebookresearch/DiT/tree/main), [MaskDiT](https://github.com/Anima-Lab/MaskDiT), and [EDM](https://github.com/NVlabs/edm). This repository is built on them.

