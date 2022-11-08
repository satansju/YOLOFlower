from PIL import Image, ExifTags
import os
from os import listdir
from os.path import isfile, join
import re
import glob
import multiprocessing.dummy as mpd
from tqdm import tqdm
from data_cleaning import clean_filename, get_series

def resizeImages(src: str, dst: str, resolution: tuple[int, int], out_ext: str = "jpg", verbose: bool = False) -> None:
    if len(resolution) != 2:
        raise ValueError("Resolution must be a tuple of length 2.")

    subdirs = [i for i in listdir(src) if not isfile(join(src, i))]

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

    with tqdm(subdirs) as t:
        for sub in t:
            t.set_description("Resizing images in " + sub)
            sub_pattern = f'{src}{sub}{os.sep}**'
            new_sub = re.sub("-", "_", get_series(clean_filename(sub)))
            # print(sub_pattern)
            files = glob.glob(sub_pattern)
            files = [re.sub("[/\\\\]+", "/", i) for i in files]
            if len(files) == 0 and verbose:
                print("No files found with pattern", sub_pattern)
            # print(files)
            # for a, b in  zip(files, [sub for i in files]):
            #     res.append(resize_img(a, b))
            pool.starmap(resize_img, zip(files, [new_sub for i in files]))
        else:
            if verbose:
                print("All done!")