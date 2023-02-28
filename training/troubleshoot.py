import sys, os
from os.path import abspath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import json
import h5py
import random
import yaml
from addict import Dict

from dataset import MammoH5Data, BalancedGroupSampler


def PFbeta(labels, predictions, beta, eps=1e-5):
    # eps is a small error term added for numerical stability
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = (ctp + eps) / (ctp + cfp + eps)
    c_recall = (ctp + eps) / (y_true_count + eps)
    print(f"Precision: {c_precision}")
    print(f"Recall: {c_recall}")
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.

def GetTestSet(reportpath, datasetpath, savepath):
    with open(reportpath, "r") as f:
        results = json.load(f)
    train_set = set()
    all_samples = set()
    for batch in results["samples"]:
        train_set.update(batch)
    with h5py.File(datasetpath, "r") as ds:
        ds.visit(lambda n: all_samples.add(n))
    test_set = all_samples - train_set
    print(len(test_set))
    test_dict = {}
    test_dict["test"] = list(test_set)
    with open(savepath, "w") as sp:
        json.dump(test_dict, sp)

def ExploreTestSet(testsetpath, metadatapath):
    with open(testsetpath, "r") as f:
        test_dict = json.load(f)
    test_img_ids = set([int(item) for item in test_dict["test"]])
    md = pd.read_json(metadatapath, orient="index")
    missing_ids = set()
    for val in test_img_ids:
        if val in md.index:
            continue
        else:
            missing_ids.add(val)
    final_ids = test_img_ids - missing_ids
    test_md = md.loc[list(final_ids), :]
    print(f"Proportion of cancer images in data: {md[md.cancer == 1].count()/float(md.size)}")
    print(f"Proportion of cancer images in test_set: {test_md[test_md.cancer == 1].count()/float(test_md.size)}")

def CreateDummyTestSet(metadatapath, savepath):
    md = pd.read_json(metadatapath, orient="index", convert_axes=False, convert_dates=False)
    cancer_ids = set(md.loc[md.cancer == 1].index)
    non_cancer_ids = set(md.index) - cancer_ids
    bal_cancer_set = set(random.sample(list(cancer_ids), int(len(cancer_ids)/2)) + random.sample(non_cancer_ids, int(len(cancer_ids)/2)))
    test_set_size = len(bal_cancer_set)
    split_size = int(0.01*test_set_size)
    low_cancer_set = set(random.sample(cancer_ids, split_size) + random.sample(non_cancer_ids, test_set_size - split_size))
    high_cancer_set = set(random.sample(non_cancer_ids, split_size) + random.sample(cancer_ids, test_set_size - split_size))
    test_dict = {
        "balanced_cancer": list(bal_cancer_set),
        "low_cancer": list(low_cancer_set),
        "high_cancer": list(high_cancer_set)
    }
    with open(savepath, "w") as f:
        json.dump(test_dict, f)

def GetF1Score(metadatapath, submissionpath, savepath):
    md = pd.read_json(metadatapath, orient="index", convert_axes=False, convert_dates=False)
    lmap = {0: "L", 1: "R"}
    pats = md.patient_id.to_list()
    lats = md.laterality.to_list()
    rlats = [lmap[val] for val in lats]
    md["prediction_id"] = ["_".join(item) for item in zip(map(str, pats), rlats)]
    all_labels = md.loc[:, ["prediction_id", "cancer"]].copy()
    all_labels.drop_duplicates(inplace=True)
    results = pd.read_csv(submissionpath)
    gt = [int(all_labels[all_labels.prediction_id == i].cancer) for i in results.prediction_id]
    results["gt"] = gt
    score = PFbeta(results["gt"], results["cancer"], 1)
    print(f"F1: {score}")
    results.to_csv(savepath, index=False)

def TroubleshootDataLoader(cfile):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')

    cfgs = Dict(yaml.load(open(abspath(cfile), "r"), Loader=yaml.Loader))
    paths = cfgs.paths
    test_cfgs = cfgs.run_params
    data_cfgs = cfgs.dataset_params

    data = MammoH5Data(device, paths.data_dest, paths.metadata_dest, data_cfgs)
    with open(paths.data_ids_dest, "r") as f:
        data_ids = json.load(f)
    for key, val in data_ids["val"].items():
        print(f"{key}: {len(val)}")
    group = test_cfgs.dataset
    labels = test_cfgs.classes
    batch_size = test_cfgs.batch_size
    test_sampler = BalancedGroupSampler(data_ids[group], labels, batch_size, shuffle=True)
    testloader = DataLoader(data, batch_size, sampler=test_sampler)
    print(len(testloader))
    return

if __name__ == "__main__":
    metadata_path = sys.argv[1]
    sub_path = sys.argv[2]
    save_path = sys.argv[3]
    GetF1Score(metadata_path, sub_path, save_path)
