import matplotlib
from matplotlib import pyplot as plt
from PIL import ImageFont, Image, ImageDraw
import numpy as np
import os
import glob

from sahi.models.base import DetectionModel
from sahi.models.yolov5 import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

from slicing.slicing import coco_to_yolo
import torch
import random
from tqdm import tqdm
import re

import imageio, exifread

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sahi_model = Yolov5DetectionModel(
    model_path='runs/train/final_model_200_epochs_NoWithered2/weights/best.pt',
    confidence_threshold=0.4,
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
    "Bud" : 0.4,
    "Flower" : 0.4,
    # "Withered" : ,
    "Immature" : 0.6,
    "Mature" : 0.7,
}


clLegend = []

for cls, col in cls_to_col.items():
    clLegend.append(matplotlib.lines.Line2D([0], [0], marker='o', color="#00000000", label=cls,
                          markerfacecolor=col, markersize=5))

ind_to_cls = {v : k for k, v in cls_to_ind.items()}

# 

# print([re.search("[^/]+\.ttf$", i)[0] for i in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') if re.search("[^/]+\.ttf$", i)])

def plot_bounding_box(src, annotation_list):
    image = Image.open(src)
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    if len(annotations) > 0:
        transformed_annotations = np.copy(annotations)
        transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
        transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 

        transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
        transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
        transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
        transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]   
          
        for ann in transformed_annotations:
            obj_cls, x0, y0, x1, y1 = ann
            plotted_image.rectangle(((x0,y0), (x1,y1)), outline=cls_to_col[ind_to_cls[int(obj_cls)]], width = 4)
          
    plt.imshow(np.array(image)) # .transpose(1, 0, 2)
    # Add the image name to the top left corner of the plot
    plt.annotate(re.search("[^/]+$", src)[0], xy=(6, 31), xytext=(6, 31), size = 5, bbox=dict(facecolor='#00000088', edgecolor = "#00000000", pad=1), color="white")
    # plt.text(0, 0, src, color="white", bbox=dict(facecolor='black', alpha=0.5, pad=1), size = 3)
    plt.axis("off")
    # plt.legend(clLegend, list(cls_to_col.keys()))

def plot_image_bbox(f, dir, split="test"):
  af = dir + "/" + "labels" + "/" + split + "/" + f + ".txt"
  with open(af, "r") as a:
      al = a.read().split("\n")[:-1]
      al = [x.split(" ") for x in al]
      al = [[float(y) for y in x ] for x in al]


  #Get the corresponding image file
  imgf = dir + "/" + "images" + "/" + split + "/" + f + ".jpg"
  if not os.path.exists(imgf):
    raise ValueError("Couldn't find file " + imgf)

  #Load the image
  image = Image.open(imgf)

  #Plot the Bounding Box
  plot_bounding_box(image, al)

matplotlib.rcParams["figure.figsize"] = [4, 3]
matplotlib.rcParams["figure.dpi"] = 500
matplotlib.rcParams["font.size"] = 5

fig = plt.figure()
wp = 0

# all_img_paths = glob.glob("../Processed/Reduced/images/**/*.jpg")

all_img_paths = ["../Processed/Reduced/images/" +  i for i in [
    "BJOR_01/BJOR_01_19_07_14_09_00.jpg",
    "NARS_04/NARS_04_000061.jpg",
    "NARS_35/NARS_35_000046.jpg",
    "NARS_36/NARS_36_000010.jpg"
]]

series = ["NARS_14"]

series_dict = {re.sub("_24H|_6H","",re.sub("-", "_", i)) : i for i in os.listdir("../Raw_data/") if not re.search("\.csv$", i)}

def sorted_images(srcs):
    dates = []
    
    for src in srcs:
        with open(src, "rb") as f:
            date = str(exifread.process_file(f)["EXIF DateTimeOriginal"])[5:10]
            date = float(re.sub(":", ".", date))
            dates.append(date)
            
    return [i for _, i in sorted(zip(dates, srcs))]
            

with tqdm([]) as pbar:
    for s in series:
        s_dir = "../Raw_data/" + series_dict[s]
        series_images = sorted_images(glob.glob(s_dir + "/**"))[::8]
    
        # assert False
        
        pbar.reset(total = len(series_images))
        
        if not os.path.exists("plots/" + s):
            os.mkdir("plots/" + s)

        for ind, unsliced in enumerate(series_images):
            pbar.update()
            image = Image.open(unsliced).reduce(4)
            test_res = get_sliced_prediction(image, sahi_model, 640, 640, .1, .1,
                                            perform_standard_pred=False, 
                                            postprocess_match_threshold=0.4,
                                            postprocess_type="NMS", verbose=0)

            test_list = []

            for obj in test_res.object_prediction_list:
                score = obj.score.value
                cls = obj.category.name
                if cls_thresh[cls] > score:
                    continue
                test_list.append(coco_to_yolo(obj.to_coco_annotation(), 1520, 855))

            test_numpy = np.array(test_list, dtype=np.float32)

            # ax = fig.add_subplot(2,4, ind + 1)

            plot_bounding_box(unsliced, test_numpy)
            plt.legend(clLegend, list(cls_to_col.keys()))
            plt.savefig("plots/" + s + "/" + str(ind) + ".jpg")
            plt.close()
        
        
        with imageio.get_writer("plots/" + s + ".mp4", mode='I', fps = 3) as writer:
            files = [i for _, i in sorted(zip([int(re.search("[0-9]+(?=\.jpg$)", j)[0]) for j in glob.glob("plots/" + s + "/*.jpg")], glob.glob("plots/" + s + "/*.jpg")))]
            
            print(files)
            
            for filename in files:
                image = imageio.imread(filename)
                writer.append_data(image)
        # plt.subplots_adjust(wspace=None, hspace=None)
        # plt.tight_layout()
        # plt.legend(clLegend, list(cls_to_col.keys()))
        # plt.savefig("plots/forBiology.jpg")
        # # plt.savefig("plots/prediction_" + str(i) + ".jpg")
        # plt.show()