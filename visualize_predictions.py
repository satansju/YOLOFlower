import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os
import glob

from sahi.models.base import DetectionModel
from sahi.models.yolov5 import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

from slicing.slicing import coco_to_yolo

sahi_model = Yolov5DetectionModel(
    model_path='runs/train/truefinal_model_200_epochs4/weights/best.pt',
    confidence_threshold=0
)

cls_to_ind = {
    "Bud" : 0,
    "Flower" : 1,
    "Withered" : 2,
    "Immature" : 3,
    "Mature" : 4,
  }

cls_to_col = {
    "Bud" : "#1B9E77",
    "Flower" : "#D95F02",
    "Withered" : "#7570B3",
    "Immature" : "#E7298A",
    "Mature" : "#66A61E",
}

clLegend = []

for cls, col in cls_to_col.items():
    clLegend.append(matplotlib.lines.Line2D([0], [0], marker='o', color="black", label=cls,
                          markerfacecolor=col, markersize=15))

ind_to_cls = {v : k for k, v in cls_to_ind.items()}

def plot_bounding_box(image, annotation_list):
    print(image.size)
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
          plotted_image.rectangle(((x0,y0), (x1,y1)), outline=cls_to_col[ind_to_cls[int(obj_cls)]], width = 2)
          
          # plotted_image.text((x0, y0 - 10), ind_to_cls[(int(obj_cls))], fill="cyan")
    
    plt.imshow(np.array(image))
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

matplotlib.rcParams["figure.figsize"] = [20,20]
matplotlib.rcParams["figure.dpi"] = 200

fig = plt.figure()
wp = 0

cls_thresh = {
    "Bud" : .3,
    "Flower" : .3,
    "Withered" : .3,
    "Immature" : .3,
    "Mature" : .3
}

for ind, test_unsliced_img in enumerate(sorted(glob.glob("../Processed/Reduced/images/NARS_21/*.jpg"))):
  if not ind % 25 == 0:
    continue
  wp += 1

  test_res = get_sliced_prediction(test_unsliced_img, sahi_model, 640, 640, .1, .1)
  
  test_list = []

  for i in test_res.object_prediction_list:
    score = i.score.value
    cls = i.category.name
    if cls_thresh[cls] > score:
      continue
    test_list.append(coco_to_yolo(i.to_coco_annotation(), 1520, 855))

  test_numpy = np.array(test_list, dtype=np.float32)
  
  ax = fig.add_subplot(2,2, wp)
  
  plot_bounding_box(Image.open(test_unsliced_img), test_numpy)

print("Number of images:", wp)

plt.subplots_adjust(wspace=None, hspace=None)
plt.tight_layout()
plt.legend(clLegend, list(cls_to_col.keys()))
plt.savefig("plots/prediction_NARS_20.jpg")
plt.show()