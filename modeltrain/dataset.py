import sys, os
from os.path import isdir, abspath
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import torch
import torchio as tio
import json
import yaml
from addict import Dict
import h5py
from torch.utils.data import Dataset, DataLoader, Sampler

#for viewing and troubleshooting
import matplotlib.pyplot as plt

class MammoH5Data(Dataset):

    def __init__(self, device, cfgs):
        super().__init__()
        self.device = device
        self.datapath =abspath(cfgs.datapath)
        self.metadata_path = abspath(cfgs.metadata_path)
        self.traintest_path = abspath(cfgs.traintest_path)
        self.metadata = None
        self.md_fext_map = {
            "json": self._ReadJson,
            "csv": self._ReadCSV
        }
        self.aug_map = defaultdict(self._AugInvalid,
            {
                "contrast_brightness": tio.RescaleIntensity(out_min_max=(0, 1),
                                                            percentiles=(0, 99.5))
            }
        )
        self.aug = cfgs.augmentations
        self.labels = cfgs.labels
        self.ReadMetadata() # reads data into self.metadata
        self.GetDataIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, key):
        """outputs a list [tuple, torch.tensor, tuple, torch.tensor]"""
        with h5py.File(self.datapath, "r") as f:
            ds: h5py.Dataset = f.get(key)
            ds_arr = np.zeros_like(ds, dtype=np.float32)
            ds.read_direct(ds_arr)
        md = self.metadata.loc[key, self.labels].to_numpy(np.float32)
        ds_aug = self.Augment(np.expand_dims(ds_arr, axis=(0, 3))).squeeze(-1)
        mdT = torch.from_numpy(md).to(self.device)
        dsT = torch.from_numpy(ds_aug).to(self.device)
        return key, dsT, mdT

    def _ReadCSV(self):
        self.metadata = pd.read_csv(self.metadata_path)
        return

    def _ReadJson(self):
        self.metadata = pd.read_json(self.metadata_path, orient="index",
                                      convert_axes=False, convert_dates=False)
        return

    def ReadMetadata(self):
        fext = self.metadata_path.split(".", 1)[-1]
        self.md_fext_map.get(fext, lambda: "Invalid file extension for metadata")()
        return

    def GetDataIds(self):
        self.img_ids = list(self.metadata.index.astype(str))
        return

    def _AugInvalid(self):
        prompt = "Invalid augmentation"
        raise Exception(prompt)

    def Augment(self, im):
        aug_list = []
        for key in self.aug:
            aug_list.append(self.aug_map[key])
        transform = tio.Compose(aug_list)
        im = transform(im)
        return im

class GroupSampler(Sampler):

    def __init__(self, group_indices: list, shuffle=False):
        self.indices = group_indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    torch.manual_seed(42)
    if  torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')

    cfilepath: str = "/Users/isaiah/GitHub/hana_mammograph/isaiah/config/train_config.yaml"
    cfgs = Dict(yaml.load(open(cfilepath, "r"), Loader=yaml.Loader))
    train_data = MammoH5Data(device, cfgs.dataset_params)
    with open(cfgs.dataset_params.traintest_path, "r") as f:
        traintestsplit = json.load(f)
    my_sampler = GroupSampler(traintestsplit["train"], shuffle=True)
    train_loader = DataLoader(train_data, batch_size=2, sampler=my_sampler)
    for i, (img_id, inp, gt) in enumerate(train_loader):
        print(inp.shape)
    # img_id, inp, gt = next(iter(train_loader))
    # im = inp.numpy().squeeze()
    # gt_vals = gt.numpy().squeeze()
    # print(img_id)
    # print(gt_vals)
    # plt.imshow(im, cmap="gray")
    # plt.show()
