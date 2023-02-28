import sys, os
import math
from copy import copy
from os.path import isdir, abspath
from collections import defaultdict
from typing import TypeVar, Optional, Iterator
import random
import numpy as np
import pandas as pd
import torch
import torchio as tio
import torch.distributed as dist
import json
import yaml
from addict import Dict
import h5py
from torch.utils.data import Dataset, DataLoader, Sampler

#for viewing and troubleshooting
import matplotlib.pyplot as plt

T_co = TypeVar('T_co', covariant=True)

class MammoH5Data(Dataset):

    def __init__(self, device, datapath, metadata_path, cfgs):
        super().__init__()
        self.device = device
        self.datapath =abspath(datapath)
        self.metadata_path = abspath(metadata_path)
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

class GroupDistSampler(Sampler[T_co]):

    def __init__(self, group_ids: list,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.group_ids = group_ids
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.group_ids) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.group_ids) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.group_ids) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            sample_ids = torch.randperm(len(self.group_ids), generator=g).tolist()
            sample_ids = random.sample(self.group_ids, len(self.group_ids))
        else:
            sample_ids = list(range(len(self.group_ids)))
            sample_ids = copy(self.group_ids)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(sample_ids)
            if padding_size <= len(sample_ids):
                sample_ids += sample_ids[:padding_size]
            else:
                sample_ids += (sample_ids * math.ceil(padding_size / len(sample_ids)))[:padding_size]
        else:
            sample_ids = sample_ids[:self.total_size]
        assert len(sample_ids) == self.total_size

        sample_ids = sample_ids[self.rank:self.total_size:self.num_replicas]
        assert len(sample_ids) == self.num_samples

        return iter(sample_ids)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

if __name__ == "__main__":
    torch.manual_seed(42)
    if  torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')

    cfilepath: str = "/Users/isaiah/GitHub/hana_mammograph/isaiah/config/config.yaml"
    cfgs = Dict(yaml.load(open(cfilepath, "r"), Loader=yaml.Loader))
    paths = cfgs.paths
    train_data = MammoH5Data(device, paths.data_dest, paths.metadata_dest,
                             paths.data_ids_dest, cfgs.dataset_params)
    with open(paths.data_ids_dest, "r") as f:
        data_ids = json.load(f)
    my_sampler = GroupSampler(data_ids["train"], shuffle=True)
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
