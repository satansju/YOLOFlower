from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import re
import glob
import multiprocessing.dummy as mpd
from tqdm import tqdm
from data_cleaning import clean_filename, get_series
import random
import numpy as np
from math import floor
from typing import Dict, List, Tuple

def equal_integer_partition(n : int, l : int):
    base = np.floor(n/l).astype(np.int32)
    left = n % l
    out = np.zeros(l, np.int32) + base
    out[random.sample(range(l), left)] += 1
    return out

def resize_images(src: str, dst: str, resolution: Tuple[int, int], out_ext: str = "jpg", num_subset : int = None, verbose: bool = False) -> None:
    if len(resolution) != 2:
        raise ValueError("Resolution must be a tuple of length 2.")

    subdirs = [i for i in listdir(src) if not isfile(join(src, i))]

    if not num_subset is None: 
        subdir_num = equal_integer_partition(num_subset, len(subdirs))
    else:
        subdir_num = np.zeros(len(subdirs), np.int64) - 1

    print(subdir_num, "=", sum(subdir_num))

    for i in subdirs:
        os.makedirs(join(dst, re.sub("-", "_", get_series(clean_filename(i)))), exist_ok=True)

    def resize_img(file: str, sub: str) -> None:
        try:
            if not re.search("\.[a-zA-Z]+$", file):
                if verbose:
                    print("No file extension found on", file)
                return 
            file_only = re.search("[a-zA-Z0-9_-]+\.[a-zA-Z]+$", file).group(0)
            file_only = clean_filename(file_only)
        except:
            raise ValueError("Could not match " + "[a-zA-Z0-9_-]+\.[a-zA-Z]+$" + " in " + file)

        if not isfile(file) and verbose: 
            print(file, "was not found!")
        
        if re.search("|".join(["\.jpg$", "\.jpeg$", "\.png$"]), file_only.lower()): 
            file_dst = dst + os.sep + sub + os.sep + re.sub("(?<=\.)[a-zA-Z]+$", out_ext, file_only)
            if not os.path.exists(file_dst):
                with Image.open(file) as image:
                    image.resize(resolution).save(file_dst)
            elif verbose:
                print(file_dst + " already exists!")
        elif verbose:
            raise RuntimeWarning("The file extension of " + file_only + " is not jpg/jpeg or png (capitalization not required).")

    pool = mpd.Pool()

    with tqdm(enumerate(zip(subdirs, subdir_num)), total = len(subdirs)) as t:
        for ind, (sub, num) in t:
            t.set_description("Resizing images in " + sub)
            sub_pattern = f'{src}{os.sep}{sub}{os.sep}**'
            new_sub = re.sub("-", "_", get_series(clean_filename(sub)))
            files = glob.glob(sub_pattern)
            files = [re.sub("[/\\\\]+", "/", i) for i in files]
            if not num == -1:
                if num > len(files):
                    subdir_num[ind] = len(files)
                    if len(subdir_num) > ind + 1:
                        left = num - len(files)
                        ran_l = min(len(subdir_num) - ind - 1, left)
                        ran = (ind + 1, ind + ran_l + 1)
                        subdir_num[range(*ran)] = subdir_num[range(*ran)] + equal_integer_partition(left, ran_l)
                    else:
                        print("Warning: Number of files in subset does not exactly match the value of 'num_subset'!")
                files = random.sample(files, subdir_num[ind]) 
            if len(files) == 0 and verbose:
                print("No files found with pattern", sub_pattern)

            pool.starmap(resize_img, zip(files, [new_sub for i in files]))
        else:
            if verbose:
                print("All done!")