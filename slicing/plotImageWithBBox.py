from PIL import Image, ImageDraw
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from slicing import SliceImageResult, SlicedImage

def plot_slice_image_result(image_list: SliceImageResult, width: int, height: int, plot_x = 4, plot_y = 4) -> None:
    cls_to_ind = {
        "Bud" : 0,
        "Flower" : 1,
        "Withered" : 2,
        "Immature" : 3,
        "Mature" : 4,
        "Gone" : 5
    }

    cls_to_col = {
        "Bud" : "orange",
        "Flower" : "magenta",
        "Withered" : "brown",
        "Immature" : "yellow",
        "Mature" : "cyan",
        "Gone" : "white"
    }

    clLegend = []

    for cls, col in cls_to_col.items():
        clLegend.append(matplotlib.lines.Line2D([0], [0], marker='o', color="black", label=cls,
                            markerfacecolor=col, markersize=15))

    ind_to_cls = {v : k for k, v in cls_to_ind.items()}

    def plot_single_image(slice: SlicedImage, axes: matplotlib.axes.Axes) -> None:
        annotation_list = slice.annotation
        image = slice.image
        # print(image)
        annotations = np.array([i[1:] for i in annotation_list])
        cls = [int(i[0]) for i in annotation_list]
        w, h = image.shape[0:2]

        plt.imshow(np.array(image))
        
        if len(annotations) != 0:
            # plotted_image = ImageDraw.Draw(Image.fromarray(image), "RGB")

            transformed_annotations = np.copy(annotations)
            transformed_annotations[:,[0,2]] = annotations[:,[0,2]] * w
            transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * h 
            
            transformed_annotations[:,0] = transformed_annotations[:,0] - (transformed_annotations[:,2] / 2)
            transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
            # transformed_annotations[:,2] = transformed_annotations[:,0] + transformed_annotations[:,2]
            # transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
            
            for ann in zip(cls, transformed_annotations):
                obj_cls, (x0, y0, x1, y1) = ann
                rect = matplotlib.patches.Rectangle(
                    xy = (x0,y0), 
                    width=x1, 
                    height = y1, 
                    edgecolor=cls_to_col[ind_to_cls[int(obj_cls)]], 
                    linewidth = 1,
                    fill = False)
                axes.add_patch(rect)
            
            # plotted_image.text((x0, y0 - 10), ind_to_cls[(int(obj_cls))], fill="cyan")
        
        plt.axis("off")

    fig = plt.figure()

    for ind, i in enumerate(image_list):
        ax = fig.add_subplot(plot_y, plot_x, ind+1)
        plot_single_image(i, ax)
    
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout()
    plt.legend(clLegend, list(cls_to_col.keys()))
    plt.show()