# Script for plotting the results of YOLOv5 training runs with different hyperparameters

# The script assumes that the results are stored in a folder called "runs" in the same directory as the script
# The script also assumes that the results are stored in folders with the following naming convention:
# 1) "baseline_model" for the baseline model
# 2) "sliced_model_D{D}_(equal|full)" for the sliced model with a downscaling factor of D, with either an equal or full dataset across values of D
# Inside each folder, the script assumes that the results are stored in a tab seperated file "class_results.csv" for the class-wise results.
# Example content of the file:
# Class	Images	Instances	P	R	mAP50	mAP50-95
# all	282	15715	0.06864547443103408	0.2338113227731236	0.09590950024430725	0.030003831360503046
# Bud	282	1248	0.03711173445828209	0.003205128205128205	0.02341673768969509	0.007253066450263105
# Flower	282	2091	0.19925937468822352	0.6389287422285987	0.3388761701290349	0.11007756003053579
# Withered	282	11533	0.10685626300866483	0.526922743431891	0.11725459340280617	0.03268853032171634
# Immature	282	526	0.0	0.0	0.0	0.0
# Mature	282	317	0.0	0.0	0.0	0.0 
#
# The script will plot the results of the baseline model and the sliced model with a downscaling factor of 2, 3, 4, and 5
# The script will plot the results of the sliced model with a downscaling factor of 2, 3, 4, and 5 with an equal dataset across values of D
# The script will plot the results of the sliced model with a downscaling factor of 2, 3, 4, and 5 with a full dataset across values of D

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the path to the folder containing the results
path = f'runs{os.sep}train'

# Set the names of the folders containing the results
baseline_model = "baseline_model"
sliced_model = "sliced_model_D{D}_{dataset}"

# Set the names of the files containing the results
results_file = "class_results.csv"

# Set the names of the classes
classes = ["Bud", "Flower", "Withered", "Immature", "Mature"]

# Set the names of the metrics
metrics = ["P", "R", "mAP50", "mAP50-95"]

# Set the names of the datasets
datasets = ["equal", "full"]

# Set the downscaling factors
downscaling_factors = [2, 2.5, 3, 4, 4.5]

# Set the colors for the different datasets
colors = ["red", "blue"]

# Set the labels for the different datasets
labels = ["Equal dataset", "Full dataset"]

# Set the labels for the different metrics
labels_metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]

# Set the labels for the different classes
labels_classes = ["Bud", "Flower", "Withered", "Immature", "Mature"]

# Set the labels for the different models
labels_models = ["Baseline model", "Sliced model"]

# Set the number of rows and columns for the plots
nrows = len(metrics)

# Set the number of columns for the plots
ncols = len(classes)

# Set the size of the plots
figsize = (ncols * 5, nrows * 5)

# Set the size of the font
fontsize = 20

# Set the size of the ticks
ticksize = 15

# Set the size of the legend
legendsize = 15

# Set the size of the title
titlesize = 25

# Set the size of the labels
labelsize = 20

# Set the size of the lines
linewidth = 3

# Import all the results by defining a function that takes a model type, downscaling factor, and dataset as input and returns the results, and then calling the function for all combinations of model type, downscaling factor, and dataset
def import_results(model, D, dataset):
    # Set the path to the results
    if model == "baseline":
        path_results = os.path.join(path, baseline_model, results_file)
    elif model == "sliced":
        path_results = os.path.join(path, sliced_model.format(D = re.sub("\.", "dot", str(D)), dataset = dataset), results_file)
    # Import the results
    results = pd.read_csv(path_results, sep = "\t", header=0)
    results["dataset"] = dataset
    results["D"] = D if D != None else 5
    results["model"] = model
    # Return the results
    return results

# Use the function to import the results
results = pd.concat([import_results(model, D, dataset) for model in ["baseline", "sliced"] for D in downscaling_factors for dataset in datasets], ignore_index=True)

# For each metric, plot downscaling factor vs. metric for each class using the dataset column to color the lines

import plotnine as p9
import datar.all as r

p_data = (
    r.as_tibble(results) >>
    r.pivot_longer(["P", "R", "mAP50", "mAP50-95"], names_to = "metric", values_to = "value")
)

for i in classes + ["all"]:
    result_plot = (
        p_data >> 
        r.filter(r.f["Class"] == i) >>
        r.mutate(Type = r.f["model"] + " " + r.f["dataset"]) >>
        r.filter(r.f["Type"].isin(["baseline equal", "sliced equal", "sliced full"])) >>
        r.mutate(Type = r.f["Type"].replace({"baseline equal": "Baseline model", "sliced equal": "Sliced model (equal dataset)", "sliced full": "Sliced model (full dataset)"})) >>
        p9.ggplot(p9.aes(x = "D", y = "value", color = "Type")) 
        + p9.geom_line(size = 2)
        + p9.theme_bw()
        + p9.labs(title = f"{i}", x = "Downscaling factor", y = "Metric value", color = "Type") 
        + p9.facet_wrap("~metric", ncol = 4, scales = "free")
        + p9.theme(
            subplots_adjust = {"wspace": 0.2, "hspace": 0.2}
        )
    )
    
    result_plot.save(f"plots{os.sep}results_{i}.png", width = 10, height = 4, dpi = 300)