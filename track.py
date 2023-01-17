# Author: Asger Svenning (2022-23)
# Description:
# This script performs object detection using a YOLOv5 model (YOLOFlower)
# and tracks the detected objects using BtyeTrack (modified by Asger Svenning).
# Acknowledgements:
# Huge thanks and credit to the ByteTrack authors for the original ByteTrack implementation (https://github.com/ifzhang/ByteTrack),
# this script would not have been possible without their work.

# Tracking modules
from tracker.byte_tracker import BYTETracker
from tracker.custom_utils import * # Custom utils for BYTETracker (Author: Asger Svenning)

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

# Progress bar module
from tqdm import tqdm

# Path to the folder where the tracking results will be saved
tracking_dir = "runs/tracking_v2/"
# Path to the folder where the images are stored
data_dir = "../Resized_dataset/"

# Initialize torch device as cuda if available, otherwise use cpu (slow)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sahi_model = Yolov5DetectionModel(
    model_path='YOLOFlower.pt', # Path to the model weights
    confidence_threshold=0.2,
    device=device,
    category_mapping={
        "0" : "Bud",
        "1" : "Flower",
        "2" : "Immature",
        "3" : "Mature"
    }
)

# Initialize the class handler, change the parameters if needed!
cls_handler = Class_handler(classes = ["Bud", "Flower", "Immature", "Mature"],
                            colors = ["#1B9E77", "#D95F02", "#E7298A", "#66A61E"],
                            thresholds = [0.4, 0.4, 0.5, 0.5])

# Initialize the tracker, change the parameters if needed!
tracker = BYTETracker(DUMMY_args(track_thresh = 0.5, match_thresh = 0.05, track_buffer = 1000, mot20 = True, min_distance = 0.01))

# Initialize the nonmax-suppression postprocessor, change the parameters if needed!
postprocess = NMSPostprocess(match_threshold = 0.9)

# Flag to enable/disable image alignment (recommended, but slow!)
alignImages = True

# Create the results folder if it does not exist
if (not os.path.exists(tracking_dir)):
    os.mkdir(tracking_dir)

# Main object detection and tracking loop
with tqdm() as t:
    all_series = Raw_data(data_dir)
    for series_i, series in enumerate(all_series): 
        with open(tracking_dir + os.sep + series + ".csv", "w") as f:
            series = ImageSeries(all_series, series, pbar = t, downscaling_factor=1)
            f.write("Frame\tDateTime\tTrackID\tStartFrame\tEndFrame\tClass\tx\ty\tw\th\n") # Write the results header
            last_image = None
            for image, dateTime in series:
                # Perform image alignment, if the image is not the first one
                if not last_image is None and alignImages:
                    try:
                        tvec = ird.similarity(lit.rgb2gray_approx(last_image[::4,::4,:]), lit.rgb2gray_approx(image[::4,::4,:]))["tvec"].round(4)
                        image = ird.transform_img(image, tvec=tvec)
                    except:
                        raise Exception("Image alignment failed", image.shape, last_image.shape)
                w, h = image.shape[:2]
                # Perform the object detection on the image using SAHI and the YOLOFlower model
                raw_predictions = get_sliced_prediction(image, sahi_model, 
                                                        640, 640, .1, .1, 
                                                        perform_standard_pred=False,
                                                        postprocess_type=None, 
                                                        postprocess_match_threshold=0, 
                                                        verbose=0)
                
                # Normalize the scores based on the class-specific detection thresholds
                for obj in raw_predictions.object_prediction_list:
                    obj.score.value = cls_handler.normalize_score(obj.category.name, obj.score.value)
                
                # Remove all objects with a score below the threshold
                raw_predictions.object_prediction_list = [i for i in raw_predictions.object_prediction_list if i.score.value]
                
                # Perform non-maximum suppression on the predictions
                raw_predictions.object_prediction_list = postprocess(raw_predictions.object_prediction_list)
                
                # Collect the predictions in a list of lists
                array_predictions = [obj.bbox.to_xyxy() + [obj.score.value] for obj in raw_predictions.object_prediction_list]
                classes = [obj.category.name for obj in raw_predictions.object_prediction_list]
                classes_ind = [cls_handler.get_index(cls) for cls in classes]
                
                array_predictions = np.array(array_predictions, dtype = np.float64)
                # Update the progress bar
                series.num_detected(len(array_predictions))
                
                # Perform tracking on the predictions
                if len(classes) != 0 and len(array_predictions) != 0:
                    tracking_predictions = tracker.update(array_predictions, classes, (w/w, h/w), (w, h))
                else:
                    tracking_predictions = []
                
                # Write the results to the results tsv file
                f.write(format_tracks(tracking_predictions, series.current_path, dateTime))
                # Save the last image for alignment in the next iteration
                last_image = image
