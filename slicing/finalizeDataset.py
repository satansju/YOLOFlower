import glob
import multiprocessing.dummy as mpd
import os
import re
import sys
from shutil import rmtree

from data_cleaning import create_yolo_annotations
from image_resizing import resize_images
from tqdm import tqdm

from slicing import slice_image

verbose_global = False

def read_directories(path : str) -> tuple[str, str, str]:
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [i[:-1] for i in lines]

    return tuple(lines)

def main(downscaling_factor : str = "4", split : str = "0.8,0.16,0.04", num_subset : str = None, verbose : str = "False") -> None:

    if not num_subset is None:
        try:
            num_subset = int(num_subset)
            if num_subset <= 0:
                raise ValueError("Argument 'num_subset' is smaller than or equal to 0.")
        except:
            raise ValueError("Argument 'num_subset' must be a positive non-zero integer.") 

    verbose = False if verbose == "False" else True if verbose == "True" else None
    if verbose is None:
        raise ValueError("Argument verbose must be one of either 'True' or 'False'")

    try:
        split = tuple([float(i) for i in split.rsplit(",")])
        print(split)
        if not len(split) == 3 or not sum(split) == 1:
            raise ValueError("Argument 'split' must contain 3 values that add to 1.")
    except:
        raise ValueError("Argument 'split' must be a string consisting of 3 numbers separated by commas and adding to 1.")

    try:
        downscaling_factor = int(downscaling_factor)
    except:
        raise ValueError("dummy")
    
    source_directory, reduced_directory, sliced_directory = read_directories("directories.txt") 

    in_resolution = (6080, 3420)
    out_resolution = tuple([int(i/downscaling_factor) for i in in_resolution])

    print("Output resolution:", out_resolution)

    if os.path.exists(reduced_directory): rmtree(reduced_directory)
    if os.path.exists(sliced_directory): rmtree(sliced_directory)

    resize_images(
        src=source_directory,
        dst=reduced_directory + os.sep + "images",
        resolution=out_resolution,
        out_ext="jpg",
        num_subset=num_subset,
        verbose=verbose)

    create_yolo_annotations(
        dir=source_directory,
        out_dir=reduced_directory,
        verbose=verbose
    )

    image_path_pattern = f'{reduced_directory}{os.sep}images{os.sep}**{os.sep}**.jpg'
    sliced_image_path_pattern = f'{sliced_directory}{os.sep}images{os.sep}**.jpg'
    annotation_path_pattern = f'{reduced_directory}{os.sep}labels{os.sep}**{os.sep}**.txt'

    origin_image_set = {re.search("[a-zA-Z0-9_]+(?=\.[a-zA-Z]{2,4}$)", i).group(0) for i in glob.iglob(image_path_pattern)}
    slice_image_set = {re.search("[a-zA-Z0-9_]+(?=(_[0-9]{1,4}){4}\.[a-zA-Z]{2,4}$)", i).group(0) for i in glob.iglob(sliced_image_path_pattern)}

    if verbose_global:
        print("Origin:", len(origin_image_set), "\nUnion:", len(origin_image_set | slice_image_set), "\nIntersection:", len(origin_image_set & slice_image_set), "\nLeft:", len(origin_image_set - slice_image_set))

    left = origin_image_set - slice_image_set

    left_image_paths = []
    for i in left:
        possible_left = glob.glob(f'{reduced_directory}{os.sep}images{os.sep}**{os.sep}{i}.jpg')
        if len(possible_left) == 0:
            continue
        left_image_paths.append(possible_left[0])

    with tqdm(left_image_paths) as t:

        def tile_one_image(reduced_file: str) -> None:
            t.set_description("Tiling " + reduced_file)

            image = re.sub("[/\\\\]+", "/", reduced_file)
            annotation = re.sub("(?<=/)images(?=/)", "labels", image)
            annotation = re.sub("\.[a-zA-Z]+$", ".txt", annotation)
            
            image_uuid = re.search("[a-zA-Z0-9_]+(?=\.txt$)", annotation).group(0) # Global unique image identifier
            uuid_parts = image_uuid.rsplit("_")
            series = "_".join(uuid_parts[:2]) # Image series
            local_id = "_".join(uuid_parts[2:]) # Series unique image identifier

            slice_image(
                image=image,
                yolo_annotation=annotation,
                output_dir=f'{sliced_directory}{os.sep}*{os.sep}{series}{os.sep}{local_id}', # + re.search("(?<=/images)/[a-zA-Z0-9_-]+(?=/)", image).group(0),
                output_file_name=image_uuid + "_bbox",
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.1,
                overlap_width_ratio=0.1,
                min_out_slice_annotations=1,
                verbose=False
            )
        
        for i in t:
            tile_one_image(i)

if __name__ == '__main__':
    # raise NotImplementedError("Train/Validation/Test splitting not implemented!")
    kwargs = {}
    for i in sys.argv[1:]:
        v = re.search("(?<==).+$", i).group(0)
        k = re.search("(?<=--)[a-zA-Z_]+(?==)", i).group(0)
        kwargs[k] = v 
    main(**kwargs)