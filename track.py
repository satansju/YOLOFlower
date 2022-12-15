from tracker.byte_tracker import BYTETracker

import matplotlib
from matplotlib import pyplot as plt
from PIL import ImageFont, Image, ImageDraw
import numpy as np
import os, sys, re, glob, exifread

from sahi.models.base import DetectionModel
from sahi.models.yolov5 import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

from slicing.slicing import coco_to_yolo
import torch
from tqdm import tqdm

from functools import reduce
import contextlib
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sahi_model = Yolov5DetectionModel(
    model_path='runs/train/final_model_200_epochs_NoWithered2/weights/best.pt',
    confidence_threshold=0.15,
    device=device
)

cls_to_ind = {
    "Bud" : 0,
    "Flower" : 1,
    # "Withered" : 2,
    "Immature" : 3,
    "Mature" : 4,
  }

cls_to_ind = {k : i for i, k in enumerate(cls_to_ind.keys())}

cls_to_col = {
    "Bud" : "#1B9E77",
    "Flower" : "#D95F02",
    # "Withered" : "#7570B3",
    "Immature" : "#E7298A",
    "Mature" : "#66A61E",
}

cls_thresh = {
    "Bud" : 0.15,
    "Flower" : 0.2,
    # "Withered" : ,
    "Immature" : 0.4,
    "Mature" : 0.55,
}

class DUMMY_args:
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self, k, v)

tracker = BYTETracker(DUMMY_args(track_thresh = 0.3, match_thresh = 0.015, track_buffer = 10, mot20 = True))


all_series = os.listdir("../Processed/Reduced/images/")
series_dict = {re.sub("_24H|_6H","",re.sub("-", "_", i)) : i for i in os.listdir("../Raw_data/") if not re.search("\.csv$", i)}
# print(all_series)
all_series = [i for i in all_series if i == "BJOR_01"]
all_series = sorted(all_series)

tracking_dir = "runs/tracking2/"

if (not os.path.exists(tracking_dir)):
    os.mkdir(tracking_dir)
    

def pbar_desc(s, si, si_max, n):
    return f"Tracking objects on images (series {s} ({si}/{si_max}), {n} objects found)"

def sorted_images(srcs):
    dates = []
    
    for src in srcs:
        with open(src, "rb") as f:
            date = str(exifread.process_file(f)["EXIF DateTimeOriginal"])[5:10]
            date = float(re.sub(":", ".", date))
            dates.append(date)
            
    return [i for _, i in sorted(zip(dates, srcs))]

series_i_max = len(all_series)
with tqdm() as t:
    for series_i, series in enumerate(all_series):            
        image_paths = glob.glob("../Raw_data/" + series_dict[series] + "/**")
        image_paths = sorted_images(image_paths)
        
        t.reset(total=len(image_paths))
        t.set_description(pbar_desc(series, series_i, series_i_max, 0))
        
        if (not os.path.exists(tracking_dir + series + "/")):
            os.mkdir(tracking_dir + series + "/")
        
        with open(tracking_dir + series + "/tracks.txt", "w") as f:
            for image_path in image_paths:
                t.update()
                image = Image.open(image_path).reduce(4)
                w, h = image.size
                raw_predictions = get_sliced_prediction(image, sahi_model, 
                                                        640, 640, .1, .1, 
                                                        perform_standard_pred=False,
                                                        postprocess_type="NMS", 
                                                        postprocess_match_threshold=0.15, 
                                                        verbose=0)
                
                array_predictions = []
                classes = []
                
                for obj in raw_predictions.object_prediction_list:
                    score = obj.score.value
                    cls = obj.category.name
                    if score < cls_thresh[cls]:
                        continue
                    classes.append(cls)
                    score = (score - cls_thresh[cls]) / (1 - cls_thresh[cls])
                    bbox = obj.bbox.to_xyxy()
                    
                    array_predictions.append(bbox + [score])
                
                array_predictions = np.array(array_predictions, dtype = np.float64)
                t.set_description(pbar_desc(series, series_i, series_i_max, len(array_predictions)))
                
                if len(array_predictions) == 0:
                    array_predictions = np.array([[0, 0, 0, 0, 0]], dtype = np.float64)
                
                if len(classes) != 0:
                    tracking_predictions = tracker.update(array_predictions, classes, (w/w, h/w), (w, h))
                else:
                    tracking_predictions = []
                
                if tracking_predictions:
                    tracks = reduce(lambda x, y : f'{x}, {y}', [i.__repr__() for i in tracking_predictions])
                else:
                    tracks = ""
                
                f.write(re.search("[^/]+$", image_path)[0] + " | " + tracks + "\n")
