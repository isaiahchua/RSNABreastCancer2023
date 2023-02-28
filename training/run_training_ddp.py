import sys, os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from os.path import abspath, dirname, join, basename, isdir
import torch
from train_ddp import Train
import argparse
import yaml
import shutil
from addict import Dict
import time

def SaveOtherFiles(cfile_src, paths):
    results_path = dirname(abspath(paths.model_ckpts_dest))
    model_id = paths.model_ckpts_dest.split(".", 1)[0][-2:]
    filename = basename(cfile_src).split(".", 1)[0] + "_" + model_id
    cfile_dest = join(results_path, filename + ".yaml")
    traintestsplit_src = paths.data_ids_dest
    traintestsplit_dest = join(results_path, filename + ".json")

    if not isdir(results_path):
        os.makedirs(results_path)
    shutil.copy(cfile_src, cfile_dest)
    shutil.copy(traintestsplit_src, traintestsplit_dest)

def PrintTimeStats(start, end):
    start_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
    end_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    time_taken = end - start
    if time_taken > 3600.:
        divisor = 3600.
        suffix = "hr"
    else:
        divisor = 60.
        suffix = "min"
    print("\n")
    print(f"Start Time: {start_s}, End Time: {end_s}, Total Time Taken: {(time_taken)/divisor:.3f} {suffix}")

def main(cfgs):
    # torch.manual_seed(42)
    train = Train(cfgs)
    start = time.time()
    train.RunDDP()
    end = time.time()
    PrintTimeStats(start, end)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Training configuration yaml file."))
    args = parser.parse_args()
    src = args.cfgs
    cfgs = Dict(yaml.load(open(abspath(args.cfgs), "r"), Loader=yaml.Loader))
    SaveOtherFiles(src, cfgs.paths)
    main(cfgs)
