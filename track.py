# Tracking modules
from tracker.byte_tracker import BYTETracker
from tracker.custom_utils import *

# Model modules
from sahi.models.yolov5 import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess.combine import NMSPostprocess

# Image alignment module
import imreg_dft as ird
import lodgepole.image_tools as lit

# Extra modules
import numpy as np
import torch
import os

from tqdm import tqdm

tracking_dir = "runs/tracking/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sahi_model = Yolov5DetectionModel(
    model_path='noWithered.pt',
    confidence_threshold=0.2,
    device=device,
    category_mapping={
        "0" : "Bud",
        "1" : "Flower",
        "2" : "Immature",
        "3" : "Mature"
    }
)
cls_handler = Class_handler(classes = ["Bud", "Flower", "Immature", "Mature"],
                            colors = ["#1B9E77", "#D95F02", "#E7298A", "#66A61E"],
                            thresholds = [0.3, 0.4, 0.5, 0.6])
tracker = BYTETracker(DUMMY_args(track_thresh = 0.6, match_thresh = 0.05, track_buffer = 100, mot20 = True, min_distance = 0.01))
postprocess = NMSPostprocess(match_threshold = 0.9)

if (not os.path.exists(tracking_dir)):
    os.mkdir(tracking_dir)

with tqdm() as t:
    all_series = Raw_data("../Raw_data/")
    for series_i, series in enumerate([all_series[0]]): #.get_series("BJOR_01")  
        with open(tracking_dir + os.sep + series + ".csv", "w") as f:
            series = ImageSeries(all_series, series, pbar = t, subsample = 1)
            f.write("Frame\tDateTime\tTrackID\tStartFrame\tEndFrame\tClass\tx\ty\tw\th\n")
            last_image = None
            for image, dateTime in series:
                # Perform image alignment, if the image is not the first one
                if not last_image is None:
                    try:
                        tvec = ird.similarity(lit.rgb2gray_approx(last_image[::4,::4,:]), lit.rgb2gray_approx(image[::4,::4,:]))["tvec"].round(4)
                        image = ird.transform_img(image, tvec=tvec)
                    except:
                        raise Exception("Image alignment failed", image.shape, last_image.shape)
                w, h = image.shape[:2]
                raw_predictions = get_sliced_prediction(image, sahi_model, 
                                                        640, 640, .1, .1, 
                                                        perform_standard_pred=False,
                                                        postprocess_type=None, 
                                                        postprocess_match_threshold=0, 
                                                        verbose=0)
                
                for obj in raw_predictions.object_prediction_list:
                    obj.score.value = cls_handler.normalize_score(obj.category.name, obj.score.value)
                
                raw_predictions.object_prediction_list = [i for i in raw_predictions.object_prediction_list if i.score.value]
                
                raw_predictions.object_prediction_list = postprocess(raw_predictions.object_prediction_list)
                array_predictions = [obj.bbox.to_xyxy() + [obj.score.value] for obj in raw_predictions.object_prediction_list]
                classes = [obj.category.name for obj in raw_predictions.object_prediction_list]
                classes_ind = [cls_handler.get_index(cls) for cls in classes]
                
                array_predictions = np.array(array_predictions, dtype = np.float64)
                series.num_detected(len(array_predictions))
                
                if len(classes) != 0 and len(array_predictions) != 0:
                    tracking_predictions = tracker.update(array_predictions, classes, (w/w, h/w), (w, h))
                else:
                    tracking_predictions = []
                
                f.write(format_tracks(tracking_predictions, series.current_path, dateTime))
                last_image = image
