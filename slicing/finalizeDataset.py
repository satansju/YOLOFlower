from image_resizing import resizeImages
from data_cleaning import create_yolo_annotations
from slicing import slice_image
import multiprocessing.dummy as mpd
import glob
import os
import re
from shutil import rmtree
from tqdm import tqdm

verbose_global = False

source_directory = "../Raw data/All_w_ann/"
reduced_directory = "../Reduced 4x"
sliced_directory = "../Sliced"

in_resolution = (6080, 3420)
downscaling_factor = 4
out_resolution = tuple([int(i/downscaling_factor) for i in in_resolution])

print("Output resolution:", out_resolution)

if os.path.exists(reduced_directory): rmtree(reduced_directory)
# if os.path.exists(reduced_directory + os.sep + "labels"): rmtree(reduced_directory + os.sep + "labels")
if os.path.exists(sliced_directory): rmtree(sliced_directory)

resizeImages(
    src=source_directory,
    dst=reduced_directory + os.sep + "images",
    resolution=out_resolution,
    out_ext="jpg",
    verbose=verbose_global)

create_yolo_annotations(
    dir=source_directory,
    out_dir=reduced_directory,
    verbose=verbose_global
)

image_path_pattern = f'{reduced_directory}{os.sep}images{os.sep}**{os.sep}**.jpg'
sliced_image_path_pattern = f'{sliced_directory}{os.sep}images{os.sep}**.jpg'
annotation_path_pattern = f'{reduced_directory}{os.sep}labels{os.sep}**{os.sep}**.txt'

# print("Image pattern:", image_path_pattern)
# print("Annotation pattern:", annotation_path_pattern)

# print([re.search("[a-zA-Z0-9_\.]+$", i).group(0) for i in glob.iglob(image_path_pattern)])
# print([re.search("[a-zA-Z0-9_\.]+$", i).group(0) for i in glob.iglob(annotation_path_pattern)])

# raise NotImplementedError("Slicing disabled for now!")

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

# glob.glob(image_path_pattern)
with tqdm(left_image_paths) as t:

    def tile_one_image(reduced_file: str) -> None:
        t.set_description("Tiling " + reduced_file)

        image = re.sub("[/\\\\]+", "/", reduced_file)
        annotation = re.sub("(?<=/)images(?=/)", "labels", image)
        annotation = re.sub("\.[a-zA-Z]+$", ".txt", annotation)

        slice_image(
            image=image,
            yolo_annotation=annotation,
            output_dir=sliced_directory, # + re.search("(?<=/images)/[a-zA-Z0-9_-]+(?=/)", image).group(0),
            output_file_name=re.search("[a-zA-Z0-9_]+(?=\.txt$)", annotation).group(0) + "_bbox",
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            min_out_slice_annotations=1,
            verbose=False
        )
    
    for i in t:
        tile_one_image(i)
    # pool = mpd.Pool()

    # pool.map(tile_one_image, t)