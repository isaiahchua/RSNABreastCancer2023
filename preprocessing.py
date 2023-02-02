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
import matplotlib.pyplot as plt
import ast
import argparse
from addict import Dict
import json
import h5py
import yaml
from utils import printProgressBarRatio

# to time script
import time

class MetadataPreprocess:

    def __init__(self, src, dest, cfgs):
        self.mdpath = abspath(src)
        self.savepath = abspath(dest)
        self.inp_md = pd.read_csv(self.mdpath)
        self.out_md = None

        if cfgs.default_value == 'na':
            self.default_value = np.nan
        else:
            self.default_value = ast.literal_eval(cfgs.default_value)

        self.age_nan = cfgs.age_nan
        self.cols = cfgs.selected_columns
        self.lmap = defaultdict(lambda: self.default_value, cfgs.laterality_map)
        self.vmap = defaultdict(lambda: self.default_value, cfgs.view_map)
        self.dmap = defaultdict(lambda: self.default_value, cfgs.density_map)
        self.dncmap = defaultdict(lambda: self.default_value, cfgs.diff_neg_case_map)
        self.smap = {'json': self._SaveJson, 'csv': self._SaveCSV}

    def GenerateMetadata(self):
        md = self.inp_md[self.cols].copy()
        if self.age_nan == "mean" and self.age_nan:
            md.age.mask(md.age.isna(), md.age.mean(), inplace=True)
        elif self.age_nan:
            md = md[md.age.notna()]
        if self.lmap:
            md.laterality = md.laterality.map(self.lmap, na_action="ignore")
        if self.vmap:
            md.view = md.view.map(self.vmap, na_action="ignore")
        if self.dmap:
            md.density = md.density.map(self.dmap, na_action="ignore")
            md.density.mask(md.density.isna(), 0, inplace=True)
        if self.dncmap:
            md.difficult_negative_case = md.difficult_negative_case.map(self.dncmap, na_action="ignore")
        md.dropna(inplace=True)
        md.set_index('image_id', inplace=True)
        self.out_md = md

    def _SaveJson(self):
        self.out_md.to_json(self.savepath, orient="index", indent=4)
        return

    def _SaveCSV(self):
        self.out_md.to_csv(self.savepath, index=False)
        return

    def Save(self):
        parentdir = dirname(self.savepath)
        if not os.path.isdir(parentdir):
            os.makedirs(parentdir)
        fext = self.savepath.split(".", 1)[-1]
        self.smap.get(fext, lambda: 'Invalid File Extension')()
        print(f"Metadata file created in {self.savepath}.")
        return

class MammoPreprocess:

    def __init__(self, source_directory, savepath, file_extension="png",
                 resolution=None, ds_ratio=3., normalize=False):
        self.src = abspath(source_directory)
        self.src_len= len(self.src) + 1
        self.savepath = abspath(savepath)
        self.res = resolution
        self.init_res = [int(n * ds_ratio) for n in self.res]
        self.fext = file_extension
        self.data_methods = {
                                         'png': self._GenerateFolderTreeDataset,
                                         'npy': self._GenerateFolderTreeDataset,
                                         'npz': self._GenerateFolderTreeDataset,
                                         'h5': self._GenerateH5Dataset
        }
        self.save_methods = {
                                         'png': self._SavePNG,
                                         'npy': self._SaveNumpy,
                                         'npz': self._SaveNumpyCompressed
        }
        self.datafiles = glob.glob(join(self.src, "**/*.dcm"),
                                                recursive=True)
        self.normit = normalize

        self.img_id = None

    def ProportionInvert(self, im, alpha=0.7):
        p = np.sum(im[im == np.max(im)])/np.prod(im.shape)
        if p > alpha:
            im = im.max() - im
        return im

    def Compress(self, im, resolution):
        if np.max(resolution) > np.max(im.shape):
            print(f"WARNING: {self.img_id} input image size is smaller than output image size.")
        else:
            end_shape = (np.asarray(resolution) * (im.shape/np.max(im.shape))).astype(np.int16)[::-1]
            im = cv2.resize(im, dsize=end_shape, interpolation=cv2.INTER_CUBIC)
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
            self.img_id = ds.InstanceNumber
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

    def _SaveNumpy(self, filepath, im):
        np.save(filepath, im)
        return

    def _SaveNumpyCompressed(self, filepath, im):
        np.savez_compressed(filepath, im)
        return

    def _SavePNG(self, filepath, im):
        plt.imsave(filepath, im, cmap="gray")
        return

    def _Save(self, filepath, im):
        self.save_methods.get(self.fext, lambda: "Invalid file extension.")(filepath, im)
        return

    def _GenerateFolderTreeDataset(self):
        for i, file in enumerate(self.datafiles):
            parentdir = dirname(file[self.src_len:])
            name = file[self.src_len:].split(".", 1)[0]
            os.makedirs(join(self.savepath, parentdir), exist_ok=True)
            im = self.ProcessDicom(file)
            sfile = os.path.join(self.savepath, name + "." + self.fext)
            self._Save(sfile, im)
            printProgressBarRatio(i + 1, len(self.datafiles), prefix="Preprocessing",
                                  suffix="Images")
        print(f"{self.savepath} created.")
        return

    def _GenerateH5Dataset(self):
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

    def GenerateDataset(self):
        self.data_methods.get(self.fext, lambda: "Invalid output dataset.")()
        return

class ProcessPath(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        if self.dest in ["src", "dest"] :
            if (values == None) or (values == ".") or (values == "./"):
                values = os.getcwd()
            if values[-1] != "/":
                values = f"{values}/"
        elif self.dest == "fext":
            if values[0] == ".":
                values = values[1:]
        setattr(namespace, self.dest, values)

class ConvertList(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        values = ast.literal_eval(values)
        if not isinstance(values, list):
            raise Exception("Resolution is not a list.")
        setattr(namespace, self.dest, values)

def main(args):
    timesheet = Dict()
    if args.cfgs != None:
        cfgs = Dict(yaml.load(open(args.cfgs, "r"), Loader=yaml.Loader))
        prep_init_start = time.time()
        paths = cfgs.paths
        pcfgs = cfgs.preprocess_params
        data_prep = MammoPreprocess(paths.data_src, paths.data_dest,
                                                      pcfgs.file_extension, pcfgs.resolution,
                                                      pcfgs.init_downsample_ratio,
                                                      pcfgs.normalization)
        prep_init_end = time.time()
        prep_init_time = prep_init_end - prep_init_start

        mcfgs = cfgs.metadata_params
        md_init_start = time.time()
        mdata_prep = MetadataPreprocess(paths.metadata_src, paths.metadata_dest,
                                        mcfgs)
        md_init_end = time.time()
        md_init_time = md_init_end - md_init_start
    else:
        prep_init_start = time.time()
        data_prep = MammoPreprocess(args.source, args.destination,
                                                      args.file_extension, args.resolution,
                                                      args.init_downsample_ratio,
                                                      args.normalization)
        prep_init_end = time.time()
        prep_init_time = prep_init_end - prep_init_start

        mcfgs = Dict(yaml.load(open(args.metadata_cfile, "r"), Loader=yaml.Loader))
        md_init_start = time.time()
        mdata_prep = MetadataPreprocess(args.metadata_src, args.metadata_dest,
                                        mcfgs)
        md_init_end = time.time()
        md_init_time = md_init_end - md_init_start

    md_proc_start = time.time()
    mdata_prep.GenerateMetadata()
    mdata_prep.Save()
    md_proc_end = time.time()
    md_proc_time = md_proc_end - md_proc_start

    prep_proc_start = time.time()
    data_prep.GenerateDataset()
    prep_proc_end = time.time()
    prep_proc_time = prep_proc_end - prep_proc_start

    timesheet.metadata.initialization = md_init_time
    timesheet.metadata.process = md_proc_time
    timesheet.preprocessing.initialization = prep_init_time
    timesheet.preprocessing.process = prep_proc_time

    if args.cfgs != None:
        with open(paths.timesheet_dest, "w") as f:
            json.dump(timesheet, f, indent=4)
        print(f"Timesheet created in {paths.timesheet_dest}.")
    else:
        with open(args.timesheet_dest, "w") as f:
            json.dump(timesheet, f, indent=4)
        print(f"Timesheet created in {args.timesheet_dest}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, metavar="PATH",
                        dest="cfgs", nargs="?", default=None,
                        help=("Configuration yaml file. "
                              "All other arguments ignored if this is passed."))
    parser.add_argument("-e", "--extension", type=str, metavar="EXT",
                        dest="file_extension", nargs="?", default=".png", action=ProcessPath,
                        help=("File extension for output images."))
    parser.add_argument("-r", "--resolution", metavar="INT,INT",
                        dest="resolution", nargs="?", default=None, action=ConvertList,
                        help=("Desired resolution of the output images."))
    parser.add_argument("-d", "--downsample-ratio", metavar="FLOAT",
                        dest="init_downsample_ratio", nargs="?", default=None,
                        help=("Ratio of initial downsampled image to final resolution."))
    parser.add_argument("-n", "--normalize", action="store_true",
                        dest="normalization",
                        help=("Normalize the image to within 0, 1. "))
    parser.add_argument("-h5", "--h5-dataset", action="store_true",
                        dest="makeh5",
                        help=("Create a hdf5 dataset."))
    parser.add_argument("source", metavar="src", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Directory to search for DICOM files recursively."))
    parser.add_argument("destination", metavar="dest", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Directory to save output images."))
    parser.add_argument("metadata_src", metavar="mds", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Filepath to metadata file."))
    parser.add_argument("metadata_dest", metavar="mdd", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Filepath to save processed metadata file."))
    parser.add_argument("metadata_cfile", metavar="mdc", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Filepath to metadata configurations file."))
    parser.add_argument("timesheet_dest", metavar="tsd", nargs="?", type=str, default=None, action=ProcessPath,
                        help=("[PATH] Filepath to save timesheet."))
    args = parser.parse_args()

    main(args)
