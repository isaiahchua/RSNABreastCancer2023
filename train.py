import os, sys
from os.path import isdir, abspath, dirname
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from modeltrain import MammoH5Data, GroupSampler
from modeltrain import DenseNet
from modeltrain import PFbeta
import json
import yaml
from addict import Dict

# ViT transfer learning model? Inception net model?

class Train:

    def __init__(self, cfgs):
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.data_cfgs = cfgs.dataset_params
        self.train_cfgs = cfgs.train_params

        if not self.train_cfgs.no_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif not self.train_cfgs.no_gpu and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device('cpu')

        self.epochs = self.train_cfgs.epochs
        self.model_path = abspath(self.train_cfgs.model_path)
        self.train_report_path = abspath(self.train_cfgs.report_path)

        self.train_report = Dict({
            "loss": [],
            "score": []
        })

        self.labels = self.data_cfgs.labels
        self.loss_weight_map = {}
        for i, key in enumerate(self.labels):
            self.loss_weight_map[key] = self.train_cfgs.loss_weights[i]
        self.best_weights = None

        self.data = MammoH5Data(self.device, self.data_cfgs)
        self.traintest_path = abspath(self.data_cfgs.traintest_path)
        with open(self.traintest_path, "r") as f:
            self.traintestsplit = Dict(json.load(f))
        self.train_sampler = GroupSampler(self.traintestsplit.train, shuffle=True)
        self.val_sampler = GroupSampler(self.traintestsplit.val, shuffle=True)
        self.batch_size = self.train_cfgs.batch_size
        self.val_size = self.train_cfgs.validation_size
        self.trainloader = DataLoader(self.data, self.batch_size, sampler=self.train_sampler)
        self.validloader = DataLoader(self.data, self.val_size, sampler=self.val_sampler)

    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)):
            os.makedirs(dirname(filepath))

    def _SaveBestModel(self):
        if self.best_weights != None:
            self._CheckMakeDirs(self.model_path)
            torch.save(self.best_weights, self.model_path)

    def SaveTrainingReport(self):
        self._CheckMakeDirs(self.train_report_path)
        with open(self.train_report_path, "w") as f:
            f.write(self.train_report)

    def GetTrainingReport(self):
        return self.train_report

    def TrainDenseNet(self):
        self.best_weights = None
        self.model = DenseNet(**self.model_cfgs)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), **cfgs.optim_params)
        met_name = "PFbeta"
        a = torch.from_numpy(np.array([[self.loss_weight_map[key] for key in self.labels]],
                                               dtype=np.float32)).to(self.device)
        c1 = nn.BCELoss(reduction="none")
        c2 = nn.L1Loss(reduction="none")
        loss = torch.tensor(0.)
        best_score = 0.
        for epoch in range(1, self.epochs + 1):

            # Training loop
            self.model.train()
            for batch, (img_id, inp, gt) in enumerate(self.trainloader):
                out = self.model(inp)
                loss = torch.sum(a[:, :4]*c1(out[:, :4], gt[:, :4])) + torch.sum(a[:, 4:]*c2(out[:, 4:], gt[:, 4:]))
                self.train_report.loss.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"batch {batch}, batch_size: {self.batch_size}, loss: {loss.item()}")

            # Validation loop;  every epoch
            self.model.eval()
            img_id, vi, vt = next(iter(self.validloader))
            preds = self.model(vi)
            sco = PFbeta(preds[:, 3], vt[:, 3], beta=0.5)
            self.train_report.score.append(sco)
            print(f"epoch: {epoch}, pred: {torch.mean(preds[:, 3]).item()}, {met_name}: {sco}")
            if sco > best_score:
                self.best_weights = self.model.state_dict()
                best_score = sco
                self._SaveBestModel()
                print("Better model found and saved")
        print(f"total epochs: {self.epochs}, final_loss: {loss.item()}, best_score: {best_score}.")
        return

if __name__ == "__main__":

    torch.manual_seed(42)
    config_file: str = "/Users/isaiah/GitHub/hana_mammograph/isaiah/config/train_config.yaml"
    cfgs = Dict(yaml.load(open(config_file, "r"), Loader=yaml.Loader))
    train = Train(cfgs)
    train.TrainDenseNet()
