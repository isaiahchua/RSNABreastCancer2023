import sys, os
from glob import glob
import json
import yaml
import numpy as np
import pandas as pd
from addict import Dict
from sklearn.model_selection import train_test_split
import ast
import argparse

def SplitData(metadata_path, savepath, test_set=False, num_samples=None, test_size=None):
    md = pd.read_json(metadata_path, orient="index", convert_axes=False, convert_dates=False)
    output = Dict()
    if num_samples == None:
        indices = md.index.to_list()
    else:
        indices = md.index.to_list()[:num_samples]
    if test_set:
        output.test = indices
    else:
        output.train, output.val = train_test_split(indices, test_size=test_size)
    with open(savepath, "w") as f:
        json.dump(output, f)

class ProcessPath(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        if self.dest in ["src", "dest"] :
            if (values == None) or (values == ".") or (values == "./"):
                values = os.getcwd()
            if values[-1] != "/":
                values = f"{values}/"
        setattr(namespace, self.dest, values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", metavar="PATH", nargs="?",
                        dest="cfgs", default="",
                        help=("YAML configuration file, all other "
                              "arguments ignored if this is passed."))
    parser.add_argument("-ts", "--test-set", action="store_true",
                        dest="test_set",
                        help=("test set data."))
    parser.add_argument("source", metavar="src", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Metadata json file."))
    parser.add_argument("destination", metavar="dest", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] File to save json file of train and test indices."))
    parser.add_argument("num_samples", metavar="n", nargs="?", type=int, default=None,
                        help=("[INT] number of samples to include in split."))
    parser.add_argument("test_size", nargs="?", type=ast.literal_eval, default=None,
                        help=("[FLOAT] Fraction of data that form the test set."))
    args = parser.parse_args()
    if args.cfgs:
        config_file = os.path.abspath(args.cfgs)
        cfgs = Dict(yaml.load(open(config_file, "r"), Loader=yaml.Loader))
        paths = cfgs.paths
        SplitData(paths.metadata_dest, paths.data_ids_dest,
             cfgs.preprocess_params.test_set,
             cfgs.preprocess_params.num_samples,
             cfgs.preprocess_params.test_size)
    else:
        SplitData(args.source, args.destination, args.test_set, args.num_samples, args.test_size)

