import argparse

import yaml

from preprocessor.preprocessor import Preprocessor
from preprocessor import ljspeech, vctk, esd


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "VCTK" in config["dataset"]:
        vctk.prepare_align(config)
    if "ESD" in config["dataset"]:
        esd.prepare_align(config)
        esd.make_meta_dict(config)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ESD/preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    main(config)
    
    
