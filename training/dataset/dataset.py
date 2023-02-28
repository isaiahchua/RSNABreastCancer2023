import sys, os
import math
from copy import copy, deepcopy
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
                                                            percentiles=(0, 99.5)),
                "flip": tio.Lambda(self._CustomFlip),
                "rotate": tio.RandomAffine(),
                "noise": tio.OneOf({tio.RandomNoise(std=(0., 0.1)): 0.75,
                                    tio.RandomBlur(std=(0., 1.)): 0.25}),
            }
        )
        self.flip_list = [0, 0, 1, 2]
        self.flip_lat = {"0.0": 1., "1.0": 0., "1": 0, "0": 1}
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
        md = self.metadata.loc[key, self.labels].copy(deep=True)
        self.flip_axis = np.random.choice(self.flip_list, 1).tolist()
        if (self.flip_axis[0] == 2) & ("laterality" in self.labels):
            md.loc["laterality"] = self.flip_lat[str(md.loc["laterality"])]
        md = md.to_numpy(np.float32)
        ds_aug = self.Augment(np.expand_dims(ds_arr, axis=(0, 3))).squeeze(-1)
        keyT = torch.tensor(int(key), dtype=torch.int64).to(self.device)
        mdT = torch.from_numpy(md).to(self.device)
        dsT = torch.from_numpy(ds_aug).to(self.device)
        return keyT, dsT, mdT

    def _CustomFlip(self, im):
        if self.flip_axis[0] == 0:
            pass
        else:
            im = torch.flip(im, self.flip_axis)
        return im

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

class BalancedGroupSampler(Sampler):

    def __init__(self, group_indices: dict, labels: list, batch_size: int,
                 shuffle=False):
        first_group = group_indices[labels[0]]
        second_group = group_indices[labels[1]]
        if len(first_group) > len(second_group):
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = True
        elif len(first_group) < len(second_group):
            self.larger_group = second_group
            self.smaller_group = first_group
            self.balance_group = True
        else:
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = False
        self.batch_size = batch_size

        padding_size = len(self.larger_group) % self.batch_size
        if padding_size <= len(self.larger_group):
            pad = random.sample(self.larger_group, padding_size)
            self.larger_group += pad
        assert len(self.larger_group) % self.batch_size == 0

        if self.balance_group:
            multiplication_factor = len(self.larger_group) // len(self.smaller_group)
            remainder = len(self.larger_group) % len(self.smaller_group)
            self.smaller_group = self.smaller_group * multiplication_factor + self.smaller_group[:remainder]
            assert len(self.smaller_group) == len(self.larger_group)

        self.shuffle = shuffle
        self.num_batches = (len(self.larger_group) + len(self.smaller_group)) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            larger_sample = random.sample(self.larger_group, len(self.larger_group))
            smaller_sample = random.sample(self.smaller_group, len(self.smaller_group))
        else:
            smaller_sample = copy(self.smaller_group)
            larger_sample = copy(self.larger_group)

        sorted_samples = self.sort_samples(larger_sample, smaller_sample)
        return iter(sorted_samples)

    def __len__(self):
        return self.num_batches

    def sort_samples(self, group1, group2):
        a = np.asarray(group1).reshape((-1, self.batch_size))
        b = np.asarray(group2).reshape((-1, self.batch_size))
        return np.concatenate((a,b), axis=1).flatten().tolist()

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
            sample_ids = random.sample(self.group_ids, len(self.group_ids))
        else:
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

class DoubleBalancedGroupDistSampler(Sampler[T_co]):

    def __init__(self, first_group: list, second_group: list,
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
        self.group1 = first_group
        self.group2 = second_group
        if len(first_group) > len(second_group):
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = True
        elif len(first_group) < len(second_group):
            self.larger_group = second_group
            self.smaller_group = first_group
            self.balance_group = True
        else:
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = False
        if self.balance_group:
            multiplication_factor = len(self.larger_group) // len(self.smaller_group)
            remainder = len(self.larger_group) % len(self.smaller_group)
            self.smaller_group = self.smaller_group * multiplication_factor + self.smaller_group[:remainder]
            assert len(self.smaller_group) == len(self.larger_group)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.larger_group) % self.num_replicas != 0:
            self.num_samples = 2 * math.ceil(
                (self.group_size - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = 2 * math.ceil(len(self.larger_group) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            larger_sample = random.sample(self.larger_group, len(self.larger_group))
            smaller_sample = random.sample(self.smaller_group, len(self.smaller_group))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size//2 - len(larger_sample)
            if padding_size <= len(larger_sample):
                smaller_sample += smaller_sample[:padding_size]
                larger_sample += larger_sample[:padding_size]
            else:
                smaller_sample += (smaller_sample * math.ceil(padding_size / len(smaller_sample)))[:padding_size]
                larger_sample += (larger_sample * math.ceil(padding_size / len(larger_sample)))[:padding_size]
        else:
            smaller_sample = smaller_sample[:self.total_size]
            larger_sample = larger_sample[:self.total_size]
        assert len(larger_sample) + len(smaller_sample) == self.total_size

        sorted_samples = self.sort_samples(larger_sample, smaller_sample)
        rank_samples = sorted_samples[self.rank:self.total_size:self.num_replicas]
        assert len(rank_samples) == self.num_samples

        return iter(rank_samples)

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

    def sort_samples(self, group1, group2):
        a = np.asarray(group1).reshape((-1, self.num_replicas))
        b = np.asarray(group2).reshape((-1, self.num_replicas))
        return np.concatenate((a,b), axis=1).flatten().tolist()

    def sort_samples2(self, group1, group2):
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        joined = np.concatenate((np.asarray(group1), np.asarray(group2)))
        rng.shuffle(joined)
        return joined.tolist()


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
