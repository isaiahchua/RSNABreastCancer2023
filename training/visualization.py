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

def PlotLoss(loss, ma):
    minloss = np.min(loss.to_numpy())
    maxloss = np.max(loss.to_numpy())
    print(f"minimum: {minloss}, maximum: {maxloss}")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(loss.index, loss["loss"], s=0.3, c="red")
    ax.plot(loss.index, ma, "-b")
    plt.show()

def MovingAvg(v, size):
    avg = np.empty_like(v)
    avg[:size] = np.mean(v[:2*size])
    avg[len(v)-2*size+1:len(v)] = np.mean(v[len(v)-2*size+1:len(v)])
    for i in range(size, len(v) - size):
        avg[i] = np.mean(v[i-size:i+size])
    return avg

def ExploreLoss(reportpath, window_size):
    # results = pd.read_json(reportpath, orient="columns")
    with open(reportpath, "r") as f:
        results = json.load(f)
    for key in results.keys():
        print(f"{key}: {len(results[key])}")
    loss = pd.DataFrame({"loss": results["loss"]})
    ma = MovingAvg(loss.loss, int(window_size))
    PlotLoss(loss, ma)

def ConfusionMatrix(df, labels):
    p = df[labels[0]]
    gt = df[labels[1]]
    tp = np.sum((p == 1) & (gt == 1)).astype(int)
    fp = np.sum(p).astype(int) - tp
    tn = np.sum((p == 0) & (gt == 0)).astype(int)
    fn = np.sum(gt).astype(int) - tp
    return tp, fp, tn ,fn

def MetricsFromCM(df: pd.DataFrame, epsilon=1.0e-6):
    tp = df["tp"].to_numpy()
    fp = df["fp"].to_numpy()
    tn = df["tn"].to_numpy()
    fn = df["fn"].to_numpy()
    tpr = (tp + epsilon)/(tp + fn + epsilon)
    fpr = (fp + epsilon)/(tp + fn + epsilon)
    precision = (tp + epsilon)/(tp + fp + epsilon)
    tnr = (tn + epsilon)/(fp + tn + epsilon)
    return tpr, fpr, precision, tnr

def Dashboard(report: pd.DataFrame, window_size: int=100):
    loss_ma = MovingAvg(report.loss, window_size)
    bacc_ma = MovingAvg(report.block_accuracy, window_size)
    lr_ma = MovingAvg(report.learning_rate, window_size)
    bcm = report.groupby("block").apply(ConfusionMatrix, labels=["predictions", "truths"])
    print(bcm)
    # check bcm works
    recall, fpr, precision, tnr = MetricsFromCM(bcm)
    f1_scores = report.groupby("epoch").mean(numeric_only=True)["f1_score"]
    fig, axs = plt.subplots(2, 3, figsize=(12,8))
    fig.suptitle("Training Log")
    # plot loss
    axs[0, 0].set_title("loss")
    axs[0, 0].set_xlabel("block")
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].scatter(report.index, report.loss, s=0.3, c="red", label="value")
    axs[0, 0].plot(report.index, loss_ma, "-b", label="moving avg")
    axs[0, 0].legend()
    # plot block acc
    axs[0, 1].set_title("block accuracy")
    axs[0, 1].set_xlabel("block")
    axs[0, 1].set_ylabel("accuracy")
    axs[0, 1].scatter(report.index, report.block_accuracy, s=0.3, c="red")
    axs[0, 1].plot(report.index, bacc_ma, "-b")
    # plot epoch acc vs mean block_acc
    axs[0, 2].set_title("epoch accuracy")
    axs[0, 2].set_xlabel("epoch")
    axs[0, 2].set_ylabel("accuracy")
    axs[0, 2].plot(eacc.index, eacc.epoch_accuracy, "-r",
                   linewidth=0.3, label="eval")
    axs[0, 2].plot(eacc.index, eacc.block_accuracy, "-g",
                   linewidth=0.3, label="train")
    axs[0, 2].legend()
    # plot epoch PFbeta
    axs[1, 0].set_title("F1 Score")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_ylabel("accuracy")
    axs[1, 0].plot(f1_scores.index, f1_scores, "-r",
                   linewidth=0.3)
    # plot Recall, Precision and TNR
    axs[1, 1].set_title("AUC")
    axs[1, 1].set_xlabel("False Positive Rate")
    axs[1, 1].set_ylabel("True Negative Rate")
    axs[1, 1].plot(fpr, tnr, "-r",
                   linewidth=0.3)
    axs[1, 1].plot([0., 1.], [0., 1.], "-k", linewidth=0.3)
    # plot learning rate
    axs[1, 2].set_title("Learning Rate")
    axs[1, 2].set_xlabel("block")
    axs[1, 2].set_ylabel("accuracy")
    axs[1, 2].scatter(report.index, report.learning_rate, s=0.3, c="red")
    axs[1, 2].plot(report.index, lr_ma, "-b")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    report_path = sys.argv[1]
    window_size = int(sys.argv[2])
    df = pd.read_csv(report_path)
    Dashboard(df, window_size)
