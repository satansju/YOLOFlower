# Tracking modules
from tracker.byte_tracker import BYTETracker
from tracker.custom_utils import *

# Model modules
from sahi.models.yolov5 import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction

# Extra modules
import numpy as np
import torch
import os

from tqdm import tqdm

tracking_dir = "runs/tracking2/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sahi_model = Yolov5DetectionModel(
    model_path='runs/train/final_model_200_epochs_NoWithered2/weights/best.pt',
    confidence_threshold=0.15,
    device=device
)
cls_handler = Class_handler(classes = ["Bud", "Flower", "Immature", "Mature"],
                            colors = ["#1B9E77", "#D95F02", "#E7298A", "#66A61E"],
                            thresholds=[0.15, 0.2, 0.4, 0.5])
tracker = BYTETracker(DUMMY_args(track_thresh = 0.15, match_thresh = 0.05, track_buffer = 100, mot20 = True, min_distance = 0.01))
c
if (not os.path.exists(tracking_dir)):
    os.mkdir(tracking_dir)

with tqdm() as t:
    all_series = Raw_data("../Raw_data/")
    for series_i, series in enumerate(all_series.get_series("NARS_14")): #.get_series("BJOR_01")           
            if (not os.path.exists(tracking_dir + series + "/")):
                os.mkdir(tracking_dir + series + "/")
            with open(tracking_dir + series + "/tracks.txt", "w") as f:
                series = ImageSeries(all_series, series, pbar = t, subsample = 4)
                for image in series:
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
                        cls = obj.category.name
                        score = cls_handler.normalize_score(cls, obj.score.value)
                        if not score:
                            continue
                        classes.append(cls)
                        bbox = obj.bbox.to_xyxy()
                        
                        array_predictions.append(bbox + [score])
                    
                    array_predictions = np.array(array_predictions, dtype = np.float64)
                    series.num_detected(len(array_predictions))
                    
                    if len(classes) != 0 and len(array_predictions) != 0:
                        tracking_predictions = tracker.update(array_predictions, classes, (w/w, h/w), (w, h))
                    else:
                        tracking_predictions = []
                    
                    f.write(series.current_path + " | " + format_tracks(tracking_predictions) + "\n")
