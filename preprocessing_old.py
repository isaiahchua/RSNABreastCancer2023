import sys, os
from collections import defaultdict
from os.path import join, abspath, dirname, basename
import glob
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import label
from skimage.measure import regionprops
import pydicom
from addict import Dict
import h5py
import yaml
from utils import printProgressBarRatio

class MetadataPreprocess:

    def __init__(self, src, dest, cfgs):
        self.mdpath = abspath(src)
        self.savepath = abspath(dest)
        self.inp_md = pd.read_csv(self.mdpath)
        self.out_md = None

        self.default_value = np.nan

        self.cols = cfgs.selected_columns
        self.lmap = defaultdict(lambda: self.default_value, cfgs.laterality_map)
        self.vmap = defaultdict(lambda: self.default_value, cfgs.view_map)

    def GenerateMetadata(self):
        md = self.inp_md[self.cols].copy()
        md.laterality = md.laterality.map(self.lmap, na_action="ignore")
        md.view = md.view.map(self.vmap, na_action="ignore")
        md.dropna(inplace=True)
        md.set_index('image_id', inplace=True)
        self.out_md = md

    def Save(self):
        parentdir = dirname(self.savepath)
        if not os.path.isdir(parentdir):
            os.makedirs(parentdir)
        self.out_md.to_json(self.savepath, orient="index", indent=4)
        print(f"Metadata file created in {self.savepath}.")
        return

class MammoPreprocess:

    def __init__(self, source_directory, savepath,
                 resolution=None, ds_ratio=3., normalize=False):
        self.src = abspath(source_directory)
        self.src_len= len(self.src) + 1
        self.savepath = abspath(savepath)
        self.res = resolution
        self.init_res = [int(n * ds_ratio) for n in self.res]
        self.datafiles = glob.glob(join(self.src, "**/*.dcm"),
                                                recursive=True)
        self.normit = normalize

    def ProportionInvert(self, im, alpha=0.7):
        p = np.sum(im[im == np.max(im)])/np.prod(im.shape)
        if p > alpha:
            im = im.max() - im
        return im

    def Compress(self, im, resolution):
        if np.max(resolution) > np.max(im.shape):
            print("WARNING: input image size is smaller than output image size.")
        else:
            end_shape = (np.asarray(resolution) * (im.shape/np.max(im.shape))).astype(np.int16)[::-1]
            im = cv2.resize(im, dsize=end_shape, interpolation=cv2.INTER_NEAREST)
        return im

    def Pad(self, im):
        h, w = im.shape
        diff = np.abs(h - w)
        if h > w:
            top, bot, left, right = [0, 0, diff // 2, diff - (diff // 2)]
        else:
            top, bot, left, right = [diff // 2, diff - (diff // 2), 0, 0]
        im_pad = cv2.copyMakeBorder(im, top, bot, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=0.)
        return im_pad

    def MinThreshold(self, im, offset=5):
        min_x = np.min(im)
        mask = np.ones_like(im, dtype=np.int8)
        mask[im < min_x + offset] = 0
        return mask

    def LargestObjCrop(self, im, mask):
        mask, _ = label(mask)
        # # ignore the first index of stats because it is the background
        _, stats = np.unique(mask, return_counts=True)
        if len(stats) > 1:
            obj_idx = np.argmax(stats[1:]) + 1
            x1,y1,x2,y2 = regionprops(1*(mask == obj_idx))[0].bbox
            h = x2-x1
            w = y2-y1
            res = im[x1:x2, y1:y2]
        else:
            res = im
        return res

    def ProcessDicom(self, file):
        ds = pydicom.dcmread(file)
        im = ds.pixel_array
        if im.max() -im.min() == 0.:
            if self.res != None:
                im = np.zeros(self.res)
        else:
            if self.res != None:
                im = self.Compress(im, self.init_res)
            im = self.ProportionInvert(im)
            mask = self.MinThreshold(im)
            im = self.LargestObjCrop(im, mask)
            if self.res != None:
                im = self.Compress(im, self.res)
            im = self.Pad(im)
            if self.normit:
                im= cv2.normalize(im, None, alpha=0, beta=1,
                                                  norm_type=cv2.NORM_MINMAX,
                                                  dtype=cv2.CV_32F)
        return im

    def GenerateDataset(self):
        parentdir = dirname(self.savepath)
        if not os.path.isdir(parentdir):
            os.makedirs(parentdir)
        hdf = h5py.File(self.savepath, "w")
        for i, file in enumerate(self.datafiles):
            name = basename(file).split(".", 1)[0]
            im = self.ProcessDicom(file)
            hdf.create_dataset(name, data=im, compression="gzip",
                               compression_opts=9)
            printProgressBarRatio(i + 1, len(self.datafiles), prefix="Preprocessing",
                                  suffix="Images")
        hdf.close()
        print(f"{self.savepath} created.")
        return

def main(args):
    cfgs = Dict(yaml.load(open(args.cfgs, "r"), Loader=yaml.Loader))
    paths = cfgs.paths
    pcfgs = cfgs.preprocess_params
    data_prep = MammoPreprocess(paths.data_src, paths.data_dest,
                                                  pcfgs.resolution,
                                                  pcfgs.init_downsample_ratio,
                                                  pcfgs.normalization)

    mcfgs = cfgs.metadata_params
    mdata_prep = MetadataPreprocess(paths.metadata_src, paths.metadata_dest,
                                    mcfgs)

    mdata_prep.GenerateMetadata()
    mdata_prep.Save()
    data_prep.GenerateDataset()

if __name__ == "__main__":
    pass
