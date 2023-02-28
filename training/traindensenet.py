import sys, os
from os.path import abspath, dirname, join, basename, isdir
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from train import Train
import argparse
import yaml
import shutil
from addict import Dict
import time

def SaveConfigFile(src, paths):
    results_path = dirname(abspath(paths.model_ckpts_dest))
    model_id = paths.model_ckpts_dest.split(".", 1)[0][-2:]
    filename = basename(src).split(".", 1)[0] + "_" + model_id + ".yaml"
    cp_path = join(results_path, filename)

    if not isdir(results_path):
        os.makedirs(results_path)
    shutil.copy(src, cp_path)

def main(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(42)
    cfgs = Dict(yaml.load(open(abspath(args.cfgs), "r"), Loader=yaml.Loader))
    src = args.cfgs
    SaveConfigFile(src, cfgs.paths)

    train = Train(cfgs)
    start = time.time()
    train.TrainDenseNet()
    end = time.time()
    print("\n")
    start_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
    end_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    time_taken = end - start
    if time_taken > 3600.:
        divisor = 3600.
        suffix = "hr"
    else:
        divisor = 60.
        suffix = "min"
    print(f"Start Time: {start_s}, End Time: {end_s}, Total Time Taken: {(time_taken)/divisor:.3f} {suffix}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Training configuration yaml file."))
    args = parser.parse_args()
    main(args)
