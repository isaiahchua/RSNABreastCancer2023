import os, sys
from os.path import isdir, abspath, dirname
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import MammoH5Data, GroupSampler
from models import DenseNet
from metrics import PFbeta
import json
from addict import Dict
from utils import printProgressBarRatio

# ViT transfer learning model? Inception net model?

class Train:

    def __init__(self, cfgs):
        self.paths = cfgs.paths
        self.model_cfgs = cfgs.model_params
        self.optimizer_cfgs = cfgs.optimizer_params
        self.data_cfgs = cfgs.dataset_params
        self.train_cfgs = cfgs.run_params

        self.model_weights = None

        self.ckpts_path = abspath(self.paths.model_ckpts_dest)
        self.model_final_path = abspath(self.paths.model_final_dest)
        self.train_report_path = abspath(self.paths.train_report_path)
        self.data_path = abspath(self.paths.data_dest)
        self.metadata_path = abspath(self.paths.metadata_dest)
        self.data_ids_path = abspath(self.paths.data_ids_dest)

        if not self.train_cfgs.no_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif not self.train_cfgs.no_gpu and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device('cpu')

        self.epochs = self.train_cfgs.epochs

        self.train_report = Dict({
            "epoch": [],
            "batch": [],
            "samples": [],
            "loss": [],
            "score": []
        })

        self.labels = self.data_cfgs.labels
        self.loss_weight_map = {}
        for i, key in enumerate(self.labels):
            self.loss_weight_map[key] = self.train_cfgs.loss_weights[i]

        self.data = MammoH5Data(self.device, self.data_path, self.metadata_path,
                                self.data_cfgs)
        with open(self.data_ids_path, "r") as f:
            self.data_ids = Dict(json.load(f))
        self.train_sampler = GroupSampler(self.data_ids.train, shuffle=True)
        self.val_sampler = GroupSampler(self.data_ids.val, shuffle=True)
        self.batch_size = self.train_cfgs.batch_size
        self.val_size = self.train_cfgs.validation_size
        self.trainloader = DataLoader(self.data, self.batch_size, sampler=self.train_sampler)
        self.validloader = DataLoader(self.data, self.val_size, sampler=self.val_sampler)

    def _CheckMakeDirs(self, filepath):
        if not isdir(dirname(filepath)):
            os.makedirs(dirname(filepath))

    def _SaveBestModel(self):
        if self.model_weights != None:
            self._CheckMakeDirs(self.ckpts_path)
            torch.save(self.model_weights, self.ckpts_path)
            print(f"Best model saved to {self.ckpts_path}.")

    def _SaveFinalModel(self):
        self._CheckMakeDirs(self.model_final_path)
        torch.save(self.model_weights, self.model_final_path)
        print(f"Final model saved to {self.model_final_path}.")

    def SaveTrainingReport(self):
        self._CheckMakeDirs(self.train_report_path)
        with open(self.train_report_path, "w") as f:
            json.dump(self.train_report, f)
        print(f"Training report saved to {self.train_report_path}.")

    def GetTrainingReport(self):
        return self.train_report

    def TrainDenseNet(self):
        self.model = DenseNet(**self.model_cfgs)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), **self.optimizer_cfgs)
        met_name = "PFbeta"
        a = torch.from_numpy(np.array([[self.loss_weight_map[key] for key in self.labels]],
                                               dtype=np.float32)).to(self.device)
        loss = torch.tensor(0.).to(self.device)
        best_score = np.nan
        batches_per_epoch = len(self.trainloader)
        cancer_p = np.nan
        for epoch in range(1, self.epochs + 1):
            self.train_report.epoch.append(epoch)

            # Training loop
            self.model.train()
            for batch, (img_id, inp, gt) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                samples = list(img_id)
                out = self.model(inp)
                c1 = F.binary_cross_entropy_with_logits(out[:, :4], gt[:, :4], reduction="none")
                c2 = F.l1_loss(out[:, 4:], gt[:, 4:], reduction="none")
                loss = torch.sum(a[:, :4]*c1) + torch.sum(a[:, 4:]*c2)
                loss.backward()
                self.optimizer.step()
                self.train_report.batch.append(batch + 1)
                self.train_report.samples.append(samples)
                self.train_report.loss.append(loss.detach().to("cpu").tolist())
                pre = f"epoch {epoch}/{self.epochs} | batch"
                suf = f"size: {self.batch_size}, loss: {loss.item():.5f}, pred: {cancer_p:.3f}, {met_name}: {best_score:.3f}"
                printProgressBarRatio(batch, batches_per_epoch, prefix=pre, suffix=suf, length=50)

            # Validation loop;  every epoch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.model.eval()
            preds = []
            labels = []
            for vbatch, (vimg_id, vi, vt) in enumerate(self.validloader):
                preds.extend(torch.sigmoid(self.model(vi))[:, 3].detach().to("cpu").tolist())
                labels.extend(vt[:, 3].detach().to("cpu").tolist())
            sco = PFbeta(labels, preds, beta=0.5)
            self.train_report.score.append(float(sco))
            if sco > best_score:
                best_score = sco
                self.model_weights = self.model.state_dict()
                self._SaveBestModel()
        self.model_weights = self.model.state_dict()
        self.SaveTrainingReport()
        self._SaveFinalModel()
        return

if __name__ == "__main__":
    pass
