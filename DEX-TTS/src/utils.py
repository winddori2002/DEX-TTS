import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import neptune
import random
import fnmatch
import json 
import pickle
import yaml
import hifigan
import bigvgan

def set_experiment(args, cfg):
    
    args = get_params(args)
    for key in args:
        cfg[key] = args[key]
    
    MakeDir(cfg.checkpoint)
    ex_name = os.path.basename(os.getcwd())
    exp_id  = len(os.listdir(cfg.checkpoint))
    
    if cfg.action == 'train':
        if cfg.resume is None:
            cfg.ex_name    = f'{ex_name}-{exp_id}'
            cfg.checkpoint = os.path.join(cfg.checkpoint, cfg.ex_name)
            MakeDir(cfg.checkpoint)
        else:
            cfg.ex_name    = f'{ex_name}-{cfg.resume}'
            cfg.checkpoint = os.path.join(cfg.checkpoint, cfg.ex_name)
    else:
        cfg.ex_name    = f'{ex_name}-{cfg.test_checkpoint}'
        cfg.checkpoint = os.path.join(cfg.checkpoint, cfg.ex_name)
        
    cfg.sample_path = os.path.join(cfg.checkpoint, 'sample')
    cfg.image_path  = os.path.join(cfg.checkpoint, 'image')
    cfg.result_path = os.path.join(cfg.checkpoint, 'result')
    cfg.eval_path   = os.path.join(cfg.checkpoint, 'eval')
    MakeDir(cfg.sample_path)
    MakeDir(cfg.image_path)
    MakeDir(cfg.result_path)
    MakeDir(cfg.eval_path)
        
    return cfg

def neptune_load(PARAMS):
    """
    logging: write your neptune account/project, api topken
    """
    neptune.init('ID/Project', api_token = 'apitoken')
    neptune.create_experiment(name=PARAMS['ex_name'], params=PARAMS)
    if PARAMS['tag'] is not None:
        neptune.append_tag(str(PARAMS['tag']))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def MakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

def get_rng_state(device):
    rng_state = {
        'rand_state'   : random.getstate(),
        'numpy_state'  : np.random.get_state(),
        'torch_state'  : torch.random.get_rng_state(),
        'cuda_state'   : torch.cuda.get_rng_state(device=device),
        'os_hash_state': str(os.environ['PYTHONHASHSEED'])
    }
    return rng_state

def seed_resume(rng_state:dict, device):
    random.setstate(rng_state['rand_state'])
    np.random.set_state(rng_state['numpy_state'])
    torch.random.set_rng_state(rng_state['torch_state'])
    torch.cuda.set_rng_state(rng_state['cuda_state'], device=device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(rng_state['os_hash_state'])

def seed_init(seed=100):
    
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 
    
def find_files(root_dir, query="*.wav", include_root_dir=True):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct={}):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, dct):
        self.__dict__ = dct
    def todict(self):
        dct = {}
        for k, v in self.items():
            if issubclass(type(v), DotDict):
                v = v.todict()
            dct[k] = v
        return dct

class Config(DotDict):

    @staticmethod
    def yaml_load(path):
        ret = yaml.safe_load(open(path, 'r', encoding='utf8'))
        assert ret is not None, f'Config file {path} is empty.'
        return Config(ret)

    @staticmethod
    def trans(inp, dep=0):
        ret = ''
        if issubclass(type(inp), dict):
            for k, v in inp.items():
                ret += f'\n{"    "*dep}{k}: {Config.trans(v, dep+1)}'
        elif issubclass(type(inp), list):
            for v in inp:
                ret += f'\n{"    "*dep}- {v}'
        else:
            ret += f'{inp}'
        return ret


    def __init__(self, dct):
        if type(dct) is str:
            dct = Config.yaml_load(dct)
        super().__init__(dct)
        try:
            self._name = dct['_name']
        except:
            self._name = 'Config'
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        ret = f'[{self._name}]'
        ret += Config.trans(self)
        #for k, v in self.items():
        #    if k[0] != '_':
        #        ret += f'\n    {k:16s}: {Config.trans(v, 2)}'
        return ret


    def _apply_config(self, config, replace=False):
        for k, v in config.items():
            self[k] = v

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
        
def get_params(args):
    
    params    = {}
    args_ref  = vars(args)
    args_keys = vars(args).keys()

    for key in args_keys:
        if '__' in key:
            continue
        else:
            temp_params = args_ref[key]
            if type(temp_params) == dict:
                params.update(temp_params)
            else:
                params[key] = temp_params            
    return params

def get_cfg_params(cfg): 
    
    params   = {}
    cfg_ref  = cfg
    cfg_keys = cfg.keys()

    for key in cfg_keys:

        temp_params = cfg_ref[key]
        if type(temp_params) == DotDict:
            params.update(temp_params)
        else:
            params[key] = temp_params            
            
    return params

def Write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def Read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
        
def Write_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def Read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def Write_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f)
        
def get_vocoder(cfg):

    if cfg.vocoder == 'hifigan':
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config  = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        
        if cfg.dataset == "LJSpeech":
            print('---LJSpeech vocoder loaded---')
            ckpt = torch.load(os.path.join(cfg.path.vocoder_path, 'generator_LJSpeech.pth.tar/generator_LJSpeech.pth.tar'), map_location=cfg.device)
        else:
            print('---Universal vocoder loaded---')
            ckpt = torch.load(os.path.join(cfg.path.vocoder_path, 'generator_universal.pth.tar/generator_universal.pth.tar'), map_location=cfg.device)
            
    else:
        with open(f"bigvgan/{cfg.vocoder}_22khz_80band/config.json", "r") as f:
            config = json.load(f)
        config  = bigvgan.AttrDict(config)
        vocoder = bigvgan.Generator(config)
        
        print('---Universal vocoder loaded---')  ## only suppor univeral, but include LJSpeech, VCTK, LibriTTS 
        ### BigVGAN-Base for default
        ckpt = torch.load(os.path.join(cfg.path.vocoder_path, 'g_05000000.zip'), map_location=cfg.device)

    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(cfg.device)

    return vocoder