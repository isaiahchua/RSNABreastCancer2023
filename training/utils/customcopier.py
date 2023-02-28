import sys, os
from os.path import join, abspath, isdir, relpath, dirname
from glob import glob
import shutil
import random
import argparse

class CustomFileCopier:

    def __init__(self, srcdir, destdir, no_of_items=None, file_extension=""):
        self.srcdir = abspath(srcdir)
        self.destdir = abspath(destdir)
        self.no_of_items = no_of_items
        self.fext = file_extension

    def _PrintProgressBar(self, iteration, total, prefix = '', suffix = '',
                         length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {iteration}/{total} {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    def _GetSubset(self, all_items):
        if self.no_of_items == None:
            sel_items = all_items
        else:
            assert isinstance(self.no_of_items, int)
            assert len(all_items) > self.no_of_items
            sel_items = random.sample(all_items, self.no_of_items)
        return sel_items

    def SampleFiles(self):
        # create list of all files
        all_src_files = glob(join(self.srcdir, "**/*" + self.fext),
                             recursive=True)

        # make capture subset of files using no_of_files
        sel_src_files = self._GetSubset(all_src_files)

        # create folder tree structure of the file subset
        for i, file in enumerate(sel_src_files):
            dest_file = join(self.destdir, relpath(file, self.srcdir))
            os.makedirs(dirname(dest_file), exist_ok=True)
            # cp files into folder tree structure
            shutil.copy2(file, dest_file)
            self._PrintProgressBar(i+1, len(sel_src_files), prefix="File")
        return

    def SampleTreeFiles(self):
        # create list of all end directories with files
        all_src_files = glob(join(self.srcdir, "**/*" + self.fext),
                             recursive=True)
        all_src_dirs = set([dirname(file) for file in all_src_files])

        # make subset of folder trees using no_of_items
        sel_src_dirs = self._GetSubset(all_src_dirs)

        # create folder tree structure and copy all files
        for i, d in enumerate(sel_src_dirs):
            end_d = join(self.destdir, relpath(d, self.srcdir))
            shutil.copytree(d, end_d)
            self._PrintProgressBar(i+1, len(sel_src_dirs), prefix="Directory")
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

def Main(args):
    copier = CustomFileCopier(args.source, args.destination, args.num, args.fext)
    if args.copy_dirs:
        copier.SampleTreeFiles()
    else:
        copier.SampleFiles()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Copy a random sample of files with their "
                     "folder tree structure to a directory")
    )
    parser.add_argument("-n", "--number", metavar="INT", dest="num",
                        nargs="?", default=None, type=int,
                        help=("Optional number of files/directories to copy "
                              "from src directory. Defaults to all files."))
    parser.add_argument("-e", "--extension", type=str, metavar="EXT",
                        dest="fext", nargs="?", default=".png", action=ProcessPath,
                        help=("Optional file extension for files to copy. NOTE: "
                              "When copying directories all files will be copied "
                              "regardless of file extension."))
    parser.add_argument("-f", "--folders", action="store_true", dest="copy_dirs",
                        help=("Whether number refers to files or folders."))
    parser.add_argument("source", metavar="src", nargs="?", type=str,
                        default=None, action=ProcessPath,
                        help=("[PATH] Directory to search for files recursively."))
    parser.add_argument("destination", metavar="dest", nargs="?", type=str,
                        default=None, action=ProcessPath,
                        help=("[PATH] Parent directory to copy files to."))
    args = parser.parse_args()
    Main(args)

