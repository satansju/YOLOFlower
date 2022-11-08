# python3 test_slicing.py output_dir="test" slice_height=400 slice_width=400 min_out_slice_annotations=1

import random
import glob
import slicing
import matplotlib as mpl
import sys
from math import ceil
import re
from plotImageWithBBox import plot_slice_image_result

def test_slice(argv):

    kwargs = {re.search(".*(?==)", i).group(0) : re.search("(?<==).*", i).group(0) for i in argv[1:]}
    
    # Change data type of kwargs containing only digits to integer
    for i in kwargs:
        if re.match("^[0-9]+$", kwargs[i]):
            kwargs[i] = int(kwargs[i])

    # Default test case parameters
    if not "overlap_height_ratio" in kwargs.keys():
        kwargs["overlap_height_ratio"] = 0.2
    if not "overlap_width_ratio" in kwargs.keys():
        kwargs["overlap_width_ratio"] = 0.2
    if not "image" in kwargs.keys() and not "yolo_annotation" in kwargs.keys():
        kwargs["image"] = iter_sample_fast(glob.iglob("../YOLO - init/Data/images/**/**.JPG"), 1)[0]
        kwargs["yolo_annotation"] = re.sub("images", "labels", re.sub("\.JPG$", ".txt", kwargs["image"]))
    elif not "image" in kwargs.keys() or not "yolo_annotation" in kwargs.keys():
        raise Exception("Either both image and yolo_annotation must be supplied or neither!")
    if not "output_file_name" in kwargs.keys() and "output_dir" in kwargs.keys():
        print(kwargs["image"])
        kwargs["output_file_name"] = re.search("[a-zA-Z0-9_]+(?=\.JPG$)",kwargs["image"]).group(0)

    result = slicing.slice_image(**kwargs)
    
    w = ceil(1280/(kwargs["slice_width"]*(1-kwargs["overlap_width_ratio"])))
    h = ceil(len(result.sliced_image_list)/w)

    # for i in result.sliced_image_list:
    #     print(i.annotation)
    
    mpl.rcParams["figure.figsize"] = [20, 20]
    plot_slice_image_result(result.sliced_image_list, kwargs["slice_width"], kwargs["slice_height"], w, h)


def iter_sample_fast(iterator, samplesize):
    results = []
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(iterator.__next__())
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results

if __name__ == "__main__":
    test_slice(sys.argv)